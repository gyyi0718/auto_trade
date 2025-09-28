# -*- coding: utf-8 -*-
# Bybit Live Trading (Unified v5, Linear Perp) — 하드 SL -1% 통합판

import os, time, math, random, threading, csv, sys
from typing import Dict, Tuple, List, Optional
import numpy as np
try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e

# ---- 출력 필터 ----
BLOCKED_KEYWORDS = ["GPU available","TPU available","IPU available","HPU available",
    "CUDA_VISIBLE_DEVICES","LOCAL_RANK","predict_dataloader",
    "FutureWarning","lightning.pytorch"]
class FilteredStdout:
    def __init__(self, s): self.s=s
    def write(self,m):
        if "[MONITOR ERR]" in m or "[SCANNER ERR]" in m: self.s.write(m); return
        if not any(k in m for k in BLOCKED_KEYWORDS): self.s.write(m)
    def flush(self): self.s.flush()
sys.stdout=FilteredStdout(sys.stdout); sys.stderr=FilteredStdout(sys.stderr)

# ===================== 설정 =====================
CATEGORY="linear"
ENTRY_MODE=os.getenv("ENTRY_MODE","model")
LEVERAGE=float(os.getenv("LEVERAGE","100"))
TAKER_FEE=float(os.getenv("TAKER_FEE","0.0006"))
MAX_OPEN=int(os.getenv("MAX_OPEN","10"))

# SL/TP
SL_ROI_PCT=float(os.getenv("SL_ROI_PCT","0.01"))  # 동적 SL
HARD_SL_PCT=0.01                                  # 하드 SL -1%
SL_ABS_USD=float(os.getenv("SL_ABS_USD","0.0"))

TP1_BPS=float(os.getenv("TP1_BPS","35.0"))
TP2_BPS=float(os.getenv("TP2_BPS","100.0"))
TP3_BPS=float(os.getenv("TP3_BPS","200.0"))
TP1_RATIO=float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO=float(os.getenv("TP2_RATIO","0.35"))
BE_EPS_BPS=float(os.getenv("BE_EPS_BPS","2.0"))
BE_AFTER_TIER=int(os.getenv("BE_AFTER_TIER","2"))
TRAIL_BPS=float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER=int(os.getenv("TRAIL_AFTER_TIER","2"))

# 필터
TREND_GAP_MIN_BPS=float(os.getenv("TREND_GAP_MIN_BPS","6.0"))
MIN_SIGMA_PCT=float(os.getenv("MIN_SIGMA_PCT","0.0003"))

# 시간/쿨다운/리스크
MAX_HOLD_SEC=int(os.getenv("MAX_HOLD_SEC","3600"))
COOLDOWN_SEC=int(os.getenv("COOLDOWN_SEC","10"))
NEG_LOSS_GLOBAL_COOLDOWN_SEC=int(os.getenv("NEG_LOSS_GLOBAL_COOLDOWN_SEC","180"))
GLOBAL_COOLDOWN_UNTIL=0.0
PEAK_EQUITY=0.0
MAX_DRAWDOWN_PCT=float(os.getenv('MAX_DRAWDOWN_PCT','0.02'))
AUTO_FLATTEN_ON_DD=(str(os.getenv('AUTO_FLATTEN_ON_DD','false')).lower() in ('1','true','yes'))
DAILY_PNL_TARGET_USD=float(os.getenv('DAILY_PNL_TARGET_USD','0.0'))
_DAILY_DAY=-1; _DAILY_PNL=0.0

RISK_SCALE_LOSS2=float(os.getenv("RISK_SCALE_LOSS2","0.85"))
RISK_SCALE_LOSS3=float(os.getenv("RISK_SCALE_LOSS3","0.70"))

RISK_PCT_OF_EQUITY=float(os.getenv("RISK_PCT_OF_EQUITY","1.0"))
ENTRY_PORTION=float(os.getenv("ENTRY_PORTION","0.75"))
MIN_NOTIONAL=float(os.getenv("MIN_NOTIONAL","25.0"))
MAX_NOTIONAL_ABS=float(os.getenv("MAX_NOTIONAL_ABS","2000.0"))
MAX_NOTIONAL_PCT_OF_EQUITY=float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY","20.0"))

USE_MARKPRICE_TRIGGER=True
FORCE_ISOLATED_MARGIN=True
SL_RETRY_TIMES=int(os.getenv("SL_RETRY_TIMES","3"))

SYMBOLS=["ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
         "LINKUSDT","ADAUSDT","SUIUSDT","1000PEPEUSDT","MNTUSDT",
         "AVAXUSDT","APTUSDT","BNBUSDT","UNIUSDT","LTCUSDT"]

TRADES_CSV=os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV=os.getenv("EQUITY_CSV","live_equity.csv")

# ===================== HTTP =====================
def _to_bool(s:str)->bool: return str(s).strip().lower() in ("1","true","y","yes")
TESTNET=_to_bool(os.getenv("BYBIT_TESTNET","False"))

API_KEY    = ""
API_SECRET = ""

client=HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt   =HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# ---- Bybit 주문 헬퍼: positionIdx 자동 재시도 ----
def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    """positionIdx 없이 1차 → 'position idx not match position mode'면 1/2로 재시도."""
    try:
        return client.place_order(**params)
    except Exception as e:
        msg = str(e)
        if "position idx not match position mode" in msg and side in ("Buy","Sell"):
            try:
                params2 = dict(params)
                params2["positionIdx"] = 1 if side=="Buy" else 2
                return client.place_order(**params2)
            except Exception as e2:
                raise e2
        raise e

# 서버 타임 보정
import time as _time
def _fetch_server_ms():
    try:
        srv=client.get_server_time()
        if "result" in srv and "timeNano" in srv["result"]:
            return int(int(srv["result"]["timeNano"])//1_000_000)
        return int(srv.get("time", int(_time.time()*1000)))
    except Exception:
        return int(_time.time()*1000)
def _measure_offset_ms():
    t0=int(_time.time()*1000); s=_fetch_server_ms(); t1=int(_time.time()*1000)
    return (s - t1) + (t1-t0)//2
TIME_OFFSET_MS=_measure_offset_ms()
print(f"[TIME] server-local diff ≈ {TIME_OFFSET_MS} ms (corrected)")
def _keep_offset():
    global TIME_OFFSET_MS
    while True:
        try: TIME_OFFSET_MS=_measure_offset_ms()
        except Exception: pass
        _time.sleep(120)
threading.Thread(target=_keep_offset,daemon=True).start()
def _dbg(*a,**k): print(*a,**k,flush=True)

# ===================== 유틸/마켓 =====================
def _round_down(x:float, step:float)->float:
    if step<=0: return x
    return math.floor(x/step)*step
def _round_to_tick(x:float, tick:float)->float:
    if tick<=0: return x
    return math.floor(x/tick)*tick

def _choose_side(model_side:Optional[str], mode:str)->Optional[str]:
    m=(mode or "model").lower()
    if m=="model":  return model_side if model_side in ("Buy","Sell") else None
    if m=="inverse":
        if model_side=="Buy": return "Sell"
        if model_side=="Sell": return "Buy"
        return None
    if m=="random": return "Buy" if random.random()<0.5 else "Sell"
    return None
# === 사이징 유틸 ===
def _round_step_down(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x / step) * step

def _round_step_up(x: float, step: float) -> float:
    if step <= 0: return x
    return math.ceil(x / step) * step

def compute_position_size(symbol: str, mid: float, equity_now: float, free_cash: float,
                          conf: float) -> tuple[float, float]:
    """return (qty, notional_usdt). 최소 주문 규격을 강제로 만족."""
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["lotSize"])
    min_qty  = float(rule["minOrderQty"])
    # 선택: USDT 최소 노셔널을 환경변수로 강제(거래소 스펙에 따라 값이 없을 수 있음)
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "10.0"))

    # 1) 예산
    use_cash_cap = equity_now * float(RISK_PCT_OF_EQUITY)            # 포지션당 최대 사용 가능 비중
    use_cash     = min(free_cash * float(ENTRY_PORTION), use_cash_cap)
    if use_cash <= 0 or mid <= 0:
        return 0.0, 0.0

    # 2) 켈리 유사 스케일
    kelly_like = float(np.clip(conf, 0.0, float(os.getenv("KELLY_MAX_ON_CASH", "0.5"))))
    use_final_cash = use_cash * max(0.05, min(kelly_like, float(os.getenv("KELLY_MAX_ON_CASH", "0.5"))))

    # 3) 레버리지 적용 → 노셔널
    lev_req = float(os.getenv("DEFAULT_LEVERAGE", "100"))
    notional_raw = use_final_cash * lev_req

    # 4) 상·하한 적용
    notional_cap = min(float(MAX_NOTIONAL_ABS), equity_now * (float(MAX_NOTIONAL_PCT_OF_EQUITY)/100.0))
    notional_usdt = max(float(MIN_NOTIONAL), min(notional_raw, notional_cap))

    # 5) 최소 노셔널 강제
    notional_usdt = max(notional_usdt, MIN_NOTIONAL_USDT)

    # 6) 수량 산출 + 규격 보정(반드시 올림으로 맞춰서 '최소 주문' 충족)
    qty = notional_usdt / mid
    qty = _round_step_up(qty, qty_step)
    if qty < min_qty:
        qty = _round_step_up(min_qty, qty_step)

    # 7) 최종 노셔널 재계산
    notional_usdt = qty * mid
    return qty, notional_usdt

def get_quote(symbol:str)->Tuple[float,float,float]:
    res=mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row=((res.get("result") or {}).get("list") or [{}])
    row=row[0] if row else {}
    f=lambda k: float(row.get(k) or 0.0)
    bid,ask,last=f("bid1Price"),f("ask1Price"),f("lastPrice")
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid,ask,mid or 0.0

def get_instrument_rule(symbol:str)->Dict[str,float]:
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst: return {"tickSize":0.0001,"lotSize":0.001,"minOrderQty":0.001}
    it=lst[0]
    tick=float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
    lot =float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq=float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    return {"tickSize":tick,"lotSize":lot,"minOrderQty":minq}

def get_wallet_equity()->float:
    try:
        wb=client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows=(wb.get("result") or {}).get("list") or []
        if not rows: return 0.0
        total=0.0
        for c in rows[0].get("coin",[]):
            if c.get("coin")=="USDT":
                total+=float(c.get("walletBalance") or 0.0)
                total+=float(c.get("unrealisedPnl") or 0.0)
        return float(total)
    except Exception:
        return 0.0

def get_positions()->Dict[str,dict]:
    res=client.get_positions(category=CATEGORY, settleCoin="USDT")
    lst=(res.get("result") or {}).get("list") or []
    out={}
    for p in lst:
        sym=p.get("symbol"); size=float(p.get("size") or 0.0)
        if not sym or size<=0: continue
        out[sym]={
            "side":p.get("side"), "qty":float(p.get("size") or 0.0),
            "entry":float(p.get("avgPrice") or 0.0),
            "liq":float(p.get("liqPrice") or 0.0),
            "positionIdx":int(p.get("positionIdx") or 0),
            "takeProfit":float(p.get("takeProfit") or 0.0),
            "stopLoss":float(p.get("stopLoss") or 0.0),
        }
    return out

def get_top_symbols_by_volume(whitelist:List[str], top_n:int=MAX_OPEN)->List[str]:
    res=mkt.get_tickers(category=CATEGORY)
    data=(res.get("result") or {}).get("list") or []
    white=set(whitelist or [])
    flt=[x for x in data if x.get("symbol") in white]
    try: flt.sort(key=lambda x: float(x.get("turnover24h") or 0.0), reverse=True)
    except Exception: pass
    return [x.get("symbol") for x in flt[:top_n]]

# ===================== 신호 =====================
def _ma_gap_bps(closes,n_fast=20,n_slow=60):
    import numpy as np
    if len(closes)<max(n_fast,n_slow): return 0.0
    ma_f=float(np.mean(closes[-n_fast:])); ma_s=float(np.mean(closes[-n_slow:]))
    if ma_s<=0: return 0.0
    return (ma_f-ma_s)/ma_s*10_000.0
def _sigma_pct(closes,lookback=120):
    import numpy as np
    if len(closes)<lookback+1: return 0.0
    x=np.array(closes[-(lookback+1):],dtype=float); r=np.diff(np.log(np.clip(x,1e-12,None)))
    return float(np.std(r))
def _ma_side_from_kline(symbol:str, interval:str="1", limit:int=120)->Optional[str]:
    try:
        k=mkt.get_kline(category=CATEGORY, symbol=symbol, interval=interval, limit=min(int(limit),1000))
        rows=(k.get("result") or {}).get("list") or []
        if len(rows)<60: return None
        closes=[float(r[4]) for r in rows][::-1]
        ma20=sum(closes[-20:])/20.0; ma60=sum(closes[-60:])/60.0
        if ma20>ma60: return "Buy"
        if ma20<ma60: return "Sell"
        return None
    except Exception as e:
        _dbg("[WARN] _ma_side_from_kline:",symbol,repr(e)); return None
def choose_entry_side(symbol:str)->Optional[str]:
    return _choose_side(_ma_side_from_kline(symbol), ENTRY_MODE)

# ===================== CSV =====================
TRADES_FIELDS=["ts","event","symbol","side","qty","entry_price","exit_price","tp1","tp2","tp3","sl_price","reason"]
EQUITY_FIELDS=["ts","equity"]
def _ensure_csv_headers():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(EQUITY_FIELDS)
def _log_trade(row:dict):
    _ensure_csv_headers()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])
def _log_equity():
    _ensure_csv_headers()
    eq=get_wallet_equity()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{eq:.6f}"])

# ===================== 주문/SL/TP =====================
def ensure_isolated_and_leverage(symbol:str):
    try:
        kw=dict(category=CATEGORY, symbol=symbol, buyLeverage=str(LEVERAGE), sellLeverage=str(LEVERAGE))
        if FORCE_ISOLATED_MARGIN: kw["tradeMode"]=1
        client.set_leverage(**kw)
    except Exception as e:
        _dbg(f"[WARN] set_leverage({symbol}) failed:",repr(e))

def place_market(symbol: str, side: str, qty: float) -> bool:
    try:
        rule = get_instrument_rule(symbol)
        qty  = _round_down(qty, rule["lotSize"])
        if qty < rule["minOrderQty"]: return False
        params = dict(
            category=CATEGORY, symbol=symbol, side=side,
            orderType="Market", qty=str(qty),
            timeInForce="IOC", reduceOnly=False,
        )
        _order_with_mode_retry(params, side=side)
        return True
    except Exception as e:
        _dbg(f"[ERR] place_market {symbol} {side} {qty}:", repr(e)); return False

def place_reduce_limit(symbol:str, side:str, qty:float, price:float)->bool:
    try:
        params = dict(
            category=CATEGORY, symbol=symbol, side=side, orderType="Limit",
            qty=str(qty), price=str(price), timeInForce="PostOnly", reduceOnly=True
        )
        _order_with_mode_retry(params, side=side)
        return True
    except Exception as e:
        _dbg(f"[WARN] place_reduce_limit {symbol} {side} {qty}@{price}:",repr(e)); return False

def compute_sl(entry:float, qty:float, side:str)->float:
    if SL_ABS_USD and SL_ABS_USD>0:
        loss=float(SL_ABS_USD)
        return (entry - (loss/max(qty,1e-12))) if side=="Buy" else (entry + (loss/max(qty,1e-12)))
    return entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)

def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _trigger_by()->str:
    return "MarkPrice" if USE_MARKPRICE_TRIGGER else "LastPrice"

def set_stop_with_retry(symbol: str, sl_price: Optional[float] = None, tp_price: Optional[float] = None, position_idx: Optional[int] = None) -> bool:
    """
    Bybit v5 set_trading_stop 안전 래퍼
    - tick 반올림
    - 1차: positionIdx 없이 시도
    - 'position idx not match position mode'면 1/2로 재시도
    - tpsl 오더 조회로 등록 검증
    """
    rule = get_instrument_rule(symbol)
    tick = float(rule.get("tickSize") or 0.0)
    if sl_price is not None: sl_price=_round_to_tick(float(sl_price), tick)
    if tp_price is not None: tp_price=_round_to_tick(float(tp_price), tick)

    def _call_once(use_idx: Optional[int]):
        params = dict(
            category=CATEGORY, symbol=symbol,
            tpslMode="Full",
            slTriggerBy=_trigger_by(), tpTriggerBy=_trigger_by(),
        )
        if sl_price is not None:
            params["stopLoss"]=f"{sl_price:.10f}".rstrip("0").rstrip(".")
            params["slOrderType"]="Market"
        if tp_price is not None:
            params["takeProfit"]=f"{tp_price:.10f}".rstrip("0").rstrip(".")
            params["tpOrderType"]="Market"
        if use_idx in (1,2): params["positionIdx"]=use_idx
        return client.set_trading_stop(**params)

    ok=False; last_err=None
    for i in range(SL_RETRY_TIMES):
        try:
            ret=_call_once(use_idx=None)
            ok=(ret.get("retCode")==0)
            _dbg(f"[SET_TPSL try#{i+1}] {symbol} code={ret.get('retCode')} msg={ret.get('retMsg')}")
            if ok: break
        except Exception as e1:
            msg=str(e1); last_err=e1
            if "position idx not match position mode" in msg:
                try:
                    use_idx=position_idx
                    if use_idx not in (1,2):
                        pos=get_positions().get(symbol) or {}
                        s=(pos.get("side") or "").strip()
                        use_idx=1 if s=="Buy" else 2 if s=="Sell" else None
                    ret=_call_once(use_idx)
                    ok=(ret.get("retCode")==0)
                    _dbg(f"[SET_TPSL retry-idx={use_idx}] {symbol} code={ret.get('retCode')} msg={ret.get('retMsg')}")
                    if ok: break
                except Exception as e2:
                    last_err=e2
            else:
                _dbg(f"[ERR] set_trading_stop {symbol}: {repr(e1)}")
        time.sleep(0.3)

    try:
        od=client.get_open_orders(category=CATEGORY, orderFilter="tpsl")
        lst=(od.get("result") or {}).get("list") or []
        have_sl=any(o.get("symbol")==symbol and str(o.get("stopOrderType")) in ("StopLoss","TrailingStop") for o in lst)
        have_tp=any(o.get("symbol")==symbol and str(o.get("stopOrderType")).startswith("TakeProfit") for o in lst)
        _dbg(f"[VERIFY SL/TP] {symbol} sl={have_sl} tp={have_tp}")
        return ok or have_sl or have_tp
    except Exception as e:
        _dbg(f"[WARN] verify tpsl {symbol}: {repr(e)}")
        if ok: return True
        if last_err: _dbg(f"[LAST_ERR] {repr(last_err)}")
        return False

# ===== 백업 강제청산 =====
def close_position_all(symbol: str)->bool:
    rule=get_instrument_rule(symbol)
    lot, minq = rule["lotSize"], rule["minOrderQty"]
    for _ in range(12):
        pos=get_positions().get(symbol)
        if not pos: return True
        side=pos["side"]; qty_live=float(pos["qty"])
        if qty_live<=0: return True
        close_side="Sell" if side=="Buy" else "Buy"
        qty_to_close=_round_down(qty_live, lot)
        if qty_to_close<minq: qty_to_close=qty_live
        try:
            params=dict(
                category=CATEGORY, symbol=symbol, side=close_side,
                orderType="Market", qty=str(qty_to_close),
                reduceOnly=True, timeInForce="IOC",
            )
            _order_with_mode_retry(params, side=close_side)
        except Exception as e:
            _dbg(f"[ERR] close_position_all {symbol}: {repr(e)}")
        time.sleep(0.4)
    return False

# ===================== HARD SL / 보정 =====================
def compute_hard_sl(entry_px: float, side: str) -> float:
    s = (side or "").lower()
    return entry_px*(1 - HARD_SL_PCT) if s in ("buy","long") else entry_px*(1 + HARD_SL_PCT)

def merge_effective_sl(entry_px: float, side: str, dyn_sl: Optional[float]) -> float:
    hard=compute_hard_sl(entry_px, side)
    if dyn_sl is None or dyn_sl != dyn_sl:  # NaN
        return hard
    return max(dyn_sl, hard) if (side or "").lower() in ("buy","long") else min(dyn_sl, hard)

SAFE_TICKS=5
def _safe_sl_after_entry(side: str, entry: float, mark: float, sl_px: float, tick: float) -> float:
    if tick<=0: return sl_px
    if side=="Buy"  and sl_px>=mark - SAFE_TICKS*tick: return mark - SAFE_TICKS*tick
    if side=="Sell" and sl_px<=mark + SAFE_TICKS*tick: return mark + SAFE_TICKS*tick
    return sl_px

def try_place_stop(symbol: str, sl_price: Optional[float], tp_price: Optional[float], position_idx: Optional[int]) -> bool:
    _,_,mark = get_quote(symbol)
    pos = get_positions().get(symbol)
    if not pos: return False
    side = pos["side"]
    rule = get_instrument_rule(symbol); eps = rule["tickSize"] or 1e-6

    if sl_price is not None:
        if side=="Buy"  and sl_price >= mark: sl_price = mark - eps
        if side=="Sell" and sl_price <= mark: sl_price = mark + eps

    if tp_price is not None:
        if side=="Buy"  and tp_price <= mark: tp_price = mark + eps
        if side=="Sell" and tp_price >= mark: tp_price = mark - eps
        if (side=="Buy" and tp_price <= pos["entry"]) or (side=="Sell" and tp_price >= pos["entry"]):
            tp_price = None

    if sl_price is not None:
        if (side=="Buy" and mark <= sl_price) or (side=="Sell" and mark >= sl_price):
            return close_position_all(symbol)

    return set_stop_with_retry(symbol, sl_price=sl_price, tp_price=tp_price, position_idx=position_idx)

def enforce_hard_stop_tick(symbol:str, mark_px:float):
    st=STATE.get(symbol)
    if not st or st.get("closing"): return
    pos=get_positions().get(symbol)
    if not pos: return
    side=st["side"]; entry=st["entry"]; qty=float(pos["qty"])
    if qty<=0: return
    sl_eff=st.get("sl_eff_px") or compute_hard_sl(entry, side)
    trigger=(mark_px<=sl_eff) if side=="Buy" else (mark_px>=sl_eff)
    if trigger:
        st["closing"]=True
        ok=close_position_all(symbol)
        _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":symbol,"side":side,
                    "qty":f"{qty:.8f}","entry_price":f"{entry:.6f}","exit_price":"",
                    "tp1":"","tp2":"","tp3":"","sl_price":f"{sl_eff:.6f}","reason":"HARD_SL_-1PCT"})
        if not ok: st["closing"]=False

# ===================== 상태/전략 =====================
STATE:Dict[str,dict]={}
COOLDOWN_UNTIL:Dict[str,float]={}
LOSS_STREAK=0; WIN_STREAK=0

def _init_state(symbol:str, side:str, entry:float, qty:float, lot_step:float):
    STATE[symbol]={"side":side,"entry":float(entry),"init_qty":float(qty),
        "tp_done":[False,False,False],"be_moved":False,
        "peak":float(entry),"trough":float(entry),
        "entry_ts":time.time(),"lot_step":float(lot_step),
        "last_qty":float(qty)}

def _mark_tp_flags_by_qty(symbol:str, live_qty:float):
    st=STATE.get(symbol,{}); init_qty=float(st.get("init_qty",0.0))
    if init_qty<=0: return
    rem=live_qty/init_qty
    if not st["tp_done"][0] and rem <= (1.0 - TP1_RATIO + 1e-8):
        st["tp_done"][0]=True
        _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":symbol,"side":st["side"],"qty":"","entry_price":"","exit_price":"","tp1":"","tp2":"","tp3":"","sl_price":"","reason":""})
    if not st["tp_done"][1] and rem <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1]=True
        _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":symbol,"side":st["side"],"qty":"","entry_price":"","exit_price":"","tp1":"","tp2":"","tp3":"","sl_price":"","reason":""})
    if rem <= 1e-9: st["tp_done"][2]=True

# === 사이징: 최소 주문 규격을 '올림'으로 맞춰 코딱지 수량 방지 ===
def _round_step_up(x: float, step: float) -> float:
    if step <= 0: return x
    return math.ceil(x / step) * step

def compute_position_size_exchange(symbol: str, mid: float, equity_now: float) -> tuple[float, float]:
    """
    return (qty, notional_usdt)
    - ENTRY_PORTION, RISK_PCT_OF_EQUITY, LEVERAGE, MIN_NOTIONAL, MAX_NOTIONAL_* 적용
    - 심볼 lot/minQty 준수. '올림'으로 최소 규격 강제
    """
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["lotSize"])
    min_qty  = float(rule["minOrderQty"])

    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "10.0"))  # 거래소가 요구하는 최소 노셔널 보정용

    if mid <= 0 or equity_now <= 0:
        return 0.0, 0.0

    # 1) 사용할 현금 예산
    use_cash_cap = equity_now * float(RISK_PCT_OF_EQUITY)
    use_cash     = min(equity_now * float(ENTRY_PORTION), use_cash_cap)

    # 2) 레버리지 반영한 노셔널
    notional_raw = use_cash * float(LEVERAGE)

    # 3) 상하한
    notional_cap = min(float(MAX_NOTIONAL_ABS), equity_now * (float(MAX_NOTIONAL_PCT_OF_EQUITY) / 100.0))
    notional_usdt = max(float(MIN_NOTIONAL), min(notional_raw, notional_cap))
    notional_usdt = max(notional_usdt, MIN_NOTIONAL_USDT)

    # 4) 수량 산출 + 규격 보정(올림)
    qty = notional_usdt / mid
    qty = _round_step_up(qty, qty_step)
    if qty < min_qty:
        qty = _round_step_up(min_qty, qty_step)

    # 최종 노셔널 갱신
    notional_usdt = qty * mid
    return qty, notional_usdt


def try_enter(symbol: str):
    global _DAILY_PNL
    # 가드
    if DAILY_PNL_TARGET_USD > 0 and _DAILY_PNL >= DAILY_PNL_TARGET_USD: return
    if time.time() < GLOBAL_COOLDOWN_UNTIL: return
    if COOLDOWN_UNTIL.get(symbol, 0.0) > time.time(): return
    if symbol in get_positions(): return

    # 트렌드/변동성 필터
    try:
        k = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=200)
        rows = (k.get("result") or {}).get("list") or []
        closes = [float(r[4]) for r in rows][::-1]
        gap_bps = _ma_gap_bps(closes, 20, 60)
        sig_pct = _sigma_pct(closes, 120)
        if gap_bps < TREND_GAP_MIN_BPS or sig_pct < MIN_SIGMA_PCT:
            return
    except Exception:
        pass

    # 엔트리 방향
    side = choose_entry_side(symbol)
    _dbg(f"[SIG] {symbol} side={side}")
    if side not in ("Buy", "Sell"): return

    # 호가
    bid, ask, mid = get_quote(symbol)
    if not mid or mid <= 0:
        _dbg(f"[SKIP] {symbol} invalid mid={mid}")
        return

    # 규격
    rule = get_instrument_rule(symbol)
    lot  = rule["lotSize"]; minq = rule["minOrderQty"]; tick = rule["tickSize"]

    # 사이징
    eq = get_wallet_equity()
    qty, notional = compute_position_size_exchange(symbol, mid=float(mid), equity_now=float(eq))
    if qty < minq:
        _dbg(f"[SKIP] {symbol} qty<{minq} (qty={qty})")
        return

    # 마진/레버리지 (Isolated)
    ensure_isolated_and_leverage(symbol)

    # 주문
    if not place_market(symbol, side, qty):
        return
    time.sleep(0.5)

    # 체결 확인
    pos2 = get_positions().get(symbol)
    if not pos2:
        _dbg(f"[WARN] {symbol} no position after market?")
        return
    qty_live  = float(pos2["qty"])
    entry_live = float(pos2["entry"])
    pos_idx = int(pos2.get("positionIdx") or 0)

    _init_state(symbol, side, entry_live, qty_live, lot)

    # SL/TP 계산
    sl_dyn = compute_sl(entry_live, qty_live, side)             # ROI 또는 ABS USD
    tp1    = tp_from_bps(entry_live, TP1_BPS, side)
    tp2    = tp_from_bps(entry_live, TP2_BPS, side)
    tp3    = tp_from_bps(entry_live, TP3_BPS, side)

    # 거래소 TPSL(마켓) 등록
    set_stop_with_retry(symbol, sl_price=_round_to_tick(sl_dyn, tick), tp_price=_round_to_tick(tp3, tick), position_idx=pos_idx)

    # 분할 TP 지정가(RO)
    close_side = "Sell" if side=="Buy" else "Buy"
    qty1 = _round_down(qty_live * TP1_RATIO, lot)
    qty2 = _round_down(qty_live * TP2_RATIO, lot)
    if qty1 > 0: place_reduce_limit(symbol, close_side, qty1, _round_to_tick(tp1, tick))
    if qty2 > 0: place_reduce_limit(symbol, close_side, qty2, _round_to_tick(tp2, tick))

    _dbg(f"[ENTRY] {symbol} {side} qty={qty_live:.6f} entry≈{entry_live:.6f} "
         f"notional≈{qty_live*entry_live:.2f} "
         f"TP1={tp1:.6f} TP2={tp2:.6f} TP3={tp3:.6f} SL={sl_dyn:.6f}")

# ===================== 관리 루프 =====================
def _update_trailing_and_be(symbol:str):
    st=STATE.get(symbol); pos=get_positions().get(symbol)
    if not st or not pos: return
    side=st["side"]; entry=st["entry"]; qty_live=float(pos["qty"])
    _,_,mid=get_quote(symbol)
    if side=="Buy": st["peak"]=max(st.get("peak",entry), mid)
    else:           st["trough"]=min(st.get("trough",entry), mid)

    if abs(qty_live - st.get("last_qty", qty_live))>1e-12:
        _mark_tp_flags_by_qty(symbol, qty_live); st["last_qty"]=qty_live

    tiers_done=sum(1 for x in st["tp_done"] if x)
    if (not st["be_moved"]) and (tiers_done>=BE_AFTER_TIER):
        eps=BE_EPS_BPS/10000.0
        be_px=entry*(1.0+eps) if side=="Buy" else entry*(1.0-eps)
        set_stop_with_retry(symbol, sl_price=be_px, position_idx=int(pos.get("positionIdx") or 0))
        st["be_moved"]=True
        _dbg(f"[MOVE->BE] {symbol} SL→{be_px:.6f}")
        _log_trade({"ts":int(time.time()),"event":"MOVE_BE","symbol":symbol,"side":side,"qty":"","entry_price":"","exit_price":"","tp1":"","tp2":"","tp3":"","sl_price":f"{be_px:.6f}","reason":""})

    if tiers_done>=max(1,TRAIL_AFTER_TIER):
        if side=="Buy":
            trail=float(st.get("peak", mid))*(1.0 - TRAIL_BPS/10000.0)
            if st["be_moved"]: trail=max(trail, entry*(1.0 + BE_EPS_BPS/10000.0))
        else:
            trail=float(st.get("trough", mid))*(1.0 + TRAIL_BPS/10000.0)
            if st["be_moved"]: trail=min(trail, entry*(1.0 - BE_EPS_BPS/10000.0))
        set_stop_with_retry(symbol, sl_price=trail, position_idx=int(pos.get("positionIdx") or 0))
        st["sl_eff_px"]=merge_effective_sl(entry, side, trail)
        _log_trade({"ts":int(time.time()),"event":"TRAIL_UPDATE","symbol":symbol,"side":side,"qty":"","entry_price":"","exit_price":"","tp1":"","tp2":"","tp3":"","sl_price":f"{trail:.6f}","reason":""})

def _check_time_stop_and_cooldowns(symbol:str):
    st=STATE.get(symbol); pos=get_positions().get(symbol)
    if not st or not pos: return
    now=time.time()
    if MAX_HOLD_SEC and (now - float(st.get("entry_ts", now)))>=MAX_HOLD_SEC:
        side=pos["side"]; qty=float(pos["qty"])
        if qty>0:
            close_side="Sell" if side=="Buy" else "Buy"
            try:
                client.place_order(category=CATEGORY, symbol=symbol, side=close_side,
                                   orderType="Market", qty=str(qty), reduceOnly=True)
                _dbg(f"[FORCE-CLOSE] {symbol} by TIMESTOP")
                _log_trade({"ts":int(time.time()),"event":"FORCE_CLOSE","symbol":symbol,"side":side,
                            "qty":f"{qty:.8f}","entry_price":"","exit_price":"","tp1":"","tp2":"","tp3":"","sl_price":"","reason":"TIMESTOP"})
            except Exception as e:
                _dbg(f"[ERR] force-close {symbol}:",repr(e))
        COOLDOWN_UNTIL[symbol]=time.time()+COOLDOWN_SEC

def monitor_loop(poll_sec:float=1.0):
    last_pos_keys=set(); last_eq_log=0.0
    while True:
        try:
            global PEAK_EQUITY,_DAILY_DAY,_DAILY_PNL,GLOBAL_COOLDOWN_UNTIL
            eq=get_wallet_equity()
            if eq>0:
                if PEAK_EQUITY<=0: PEAK_EQUITY=eq
                PEAK_EQUITY=max(PEAK_EQUITY, eq)
                if PEAK_EQUITY>0 and (PEAK_EQUITY - eq)/PEAK_EQUITY >= MAX_DRAWDOWN_PCT:
                    GLOBAL_COOLDOWN_UNTIL=time.time()+max(NEG_LOSS_GLOBAL_COOLDOWN_SEC,180)
                    if AUTO_FLATTEN_ON_DD:
                        for s,p in get_positions().items():
                            try:
                                side=p['side']; qty=float(p['qty'])
                                if qty>0:
                                    client.place_order(category=CATEGORY, symbol=s, side=('Sell' if side=='Buy' else 'Buy'),
                                                       orderType='Market', qty=str(qty), reduceOnly=True)
                            except Exception: pass
            if time.time()-last_eq_log>=10:
                _log_equity(); last_eq_log=time.time()

            day_now=time.gmtime().tm_day if False else time.gmtime().tm_yday
            if _DAILY_DAY!=day_now: _DAILY_DAY=day_now; _DAILY_PNL=0.0

            pos=get_positions(); pos_keys=set(pos.keys())

            closed_syms=last_pos_keys - pos_keys
            for sym in closed_syms:
                STATE.pop(sym, None)
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _dbg(f"[EXIT DETECTED] {sym} closed")
                _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":"","qty":"","entry_price":"","exit_price":"","tp1":"","tp2":"","tp3":"","sl_price":"","reason":"CLOSED"})

            for sym in pos_keys:
                _,_,mid=get_quote(sym)
                if mid: enforce_hard_stop_tick(sym, mid)
                _update_trailing_and_be(sym)
                _check_time_stop_and_cooldowns(sym)

            last_pos_keys=pos_keys
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            _dbg("[MONITOR] stopped by user"); break
        except Exception as e:
            _dbg("[MONITOR ERR]",repr(e)); time.sleep(2.0)

# ===================== 스캐너 =====================
def scanner_loop(iter_delay:float=2.5):
    while True:
        try:
            if time.time()<GLOBAL_COOLDOWN_UNTIL: time.sleep(iter_delay); continue
            syms=get_top_symbols_by_volume(SYMBOLS, top_n=min(MAX_OPEN,len(SYMBOLS)))
            if len(get_positions())>=MAX_OPEN: time.sleep(iter_delay); continue
            for s in syms:
                if len(get_positions())>=MAX_OPEN: break
                try_enter(s)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            _dbg("[SCANNER] stopped by user"); break
        except Exception as e:
            _dbg("[SCANNER ERR]",repr(e)); time.sleep(2.0)

# ===================== 실행 =====================
def main():
    print(f"[START] Bybit Live Trading MODE={ENTRY_MODE} TESTNET={TESTNET} (Isolated={FORCE_ISOLATED_MARGIN}, MarkTrigger={USE_MARKPRICE_TRIGGER})")
    _ensure_csv_headers()
    t1=threading.Thread(target=monitor_loop,args=(1.0,),daemon=True)
    t2=threading.Thread(target=scanner_loop,args=(2.5,),daemon=True)
    t1.start(); t2.start()
    while True: time.sleep(60)

if __name__=="__main__":
    main()
