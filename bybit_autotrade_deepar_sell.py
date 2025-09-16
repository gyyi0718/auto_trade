# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — PAPER 호환 모드
- PAPER_COMPAT=1 이면 bybit_test_deepar.py와 거의 동일한 가정으로 동작
  * 실행 모드: taker 체결 고정, SLIPPAGE_BPS_TAKER=1.0, MAKER_FEE=0
  * 최소 기대 ROI 필터 비활성화
  * 포지션 사이징: notional = equity * ENTRY_EQUITY_PCT * leverage
                  cap_abs = MAX_NOTIONAL_ABS * leverage
                  cap_pct = equity * leverage * (MAX_NOTIONAL_PCT_OF_EQUITY/100)
  * TP1/TP2/TP3, BE, 트레일, 보유시간 동일
- PAPER_COMPAT=0 이면 기존 실거래 확장 기능 그대로 사용

Env vars:
  BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET=0|1
  ENTRY_MODE=model|inverse|random
  LEVERAGE=10
  TAKER_FEE=0.0006
  MAKER_FEE=0.0
  EXECUTION_MODE=maker|taker
  EXIT_EXEC_MODE=maker|taker
  SLIPPAGE_BPS_TAKER=1.0
  SLIPPAGE_BPS_MAKER=0.0
  MIN_EXPECTED_ROI_BPS=20.0
  WAIT_FOR_MAKER_FILL_SEC=30
  CANCEL_UNFILLED_MAKER=1
  MAX_OPEN=5
  SL_ROI_PCT=0.01
  TP1_BPS=50.0  TP2_BPS=100.0  TP3_BPS=200.0
  TP1_RATIO=0.40 TP2_RATIO=0.35
  BE_EPS_BPS=2.0 BE_AFTER_TIER=2 TRAIL_BPS=50.0 TRAIL_AFTER_TIER=2
  MAX_HOLD_SEC=3600  COOLDOWN_SEC=10
  ENTRY_EQUITY_PCT=0.20
  REENTRY_ON_DIP=1  REENTRY_PCT=0.20  REENTRY_SIZE_PCT=0.20
  MIN_NOTIONAL_USDT=10.0
  MAX_NOTIONAL_ABS=2000.0
  MAX_NOTIONAL_PCT_OF_EQUITY=20.0
  PAPER_COMPAT=1   ← 페이퍼 호환 토글
"""

import os, time, math, csv
from typing import Optional, Dict, Tuple, List

try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e

# ===== env =====
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")
LEVERAGE   = float(os.getenv("LEVERAGE", "100"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "taker").lower()
EXIT_EXEC_MODE  = os.getenv("EXIT_EXEC_MODE", "taker").lower()
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")
MAX_OPEN   = int(os.getenv("MAX_OPEN", "10"))

SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
TP1_BPS = float(os.getenv("TP1_BPS","50.0"))
TP2_BPS = float(os.getenv("TP2_BPS","100.0"))
TP3_BPS = float(os.getenv("TP3_BPS","200.0"))
TP1_RATIO = float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.35"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
BE_AFTER_TIER = int(os.getenv("BE_AFTER_TIER","2"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))

MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","10"))

ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))
REENTRY_ON_DIP   = str(os.getenv("REENTRY_ON_DIP","1")).lower() in ("1","y","yes","true")
REENTRY_PCT      = float(os.getenv("REENTRY_PCT","0.20"))
REENTRY_SIZE_PCT = float(os.getenv("REENTRY_SIZE_PCT","0.20"))

MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT","10.0"))
MAX_NOTIONAL_ABS = float(os.getenv("MAX_NOTIONAL_ABS","2000.0"))
MAX_NOTIONAL_PCT_OF_EQUITY = float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY","20.0"))

PAPER_COMPAT = str(os.getenv("PAPER_COMPAT","1")).lower() in ("1","y","yes","true")
# TP/SL을 레버리지와 무관하게 고정(bps)로 계산
FIXED_TP_SL = str(os.getenv("FIXED_TP_SL", "1")).lower() in ("1","y","yes","true")

# PAPER 호환 모드 강제 세팅
if PAPER_COMPAT:
    EXECUTION_MODE = "taker"
    EXIT_EXEC_MODE = "taker"
    SLIPPAGE_BPS_TAKER = 1.0
    SLIPPAGE_BPS_MAKER = 0.0
    MAKER_FEE = 0.0
    # 페이퍼처럼 진입 기대 ROI 필터 제거
    MIN_EXPECTED_ROI_BPS = -1.0

TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]

SYMBOLS = [
    "BTCUSDT", "ETHUSDT","SOLUSDT"
]

SYMBOLS2 = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]

TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = ""
API_SECRET = ""
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

USE_EV = str(os.getenv("USE_EV", "0")).lower() in ("1","y","yes","true")
VERBOSE = str(os.getenv("VERBOSE", "1")).lower() in ("1","y","yes","true")
# === absolute PnL/시간기반 청산 설정 ===
ABS_TP_USD = float(os.getenv("ABS_TP_USD", "1.0"))   # 수수료 제외 절대 이익 달러
PRED_HORIZ_MIN_DEFAULT = int(os.getenv("PRED_HORIZ_MIN", "1"))  # 기본 예측 지평(분)

# 포지션별 청산 데드라인 추적
POSITION_DEADLINE = {}  # {symbol: epoch_seconds}

def get_signal_horizon_min(symbol: str) -> int:
    """
    심볼별 예측 지평(분). 모델이 1분/3분 등으로 다르면 여기서 반환.
    필요하면 choose_entry에서 horizon을 저장하도록 확장.
    """
    return PRED_HORIZ_MIN_DEFAULT

def unrealized_pnl_usd(side: str, qty: float, entry: float, mark: float) -> float:
    """
    선형(USDT) Perp: PnL = (mark-entry)*qty (롱), (entry-mark)*qty (숏)
    수수료 제외 원시 PnL.
    """
    if side == "Buy":
        return (mark - entry) * qty
    else:
        return (entry - mark) * qty

def ev_long_short_from_samples(p0: float,
                               samples,            # (M,H) or None
                               tp_bps=(50.0,100.0,200.0),
                               sl_frac=0.01,
                               fee_taker=0.0006,
                               slip_bps=1.0):
    if samples is None:
        return None, None, 0.0, 0.0
    M, H = samples.shape
    tp = min(tp_bps)/1e4
    sl = float(sl_frac)
    cost = 2*fee_taker + 2*(slip_bps/1e4)
    LW=LL=SW=SL=0
    up  = p0*(1+tp); dn  = p0*(1-sl)
    upS = p0*(1+sl); dnS = p0*(1-tp)
    for path in samples:
        i_tp = next((i for i,p in enumerate(path) if p>=up), None)
        i_sl = next((i for i,p in enumerate(path) if p<=dn), None)
        if i_tp is not None and (i_sl is None or i_tp<i_sl): LW+=1
        elif i_sl is not None and (i_tp is None or i_sl<i_tp): LL+=1
        i_tpS = next((i for i,p in enumerate(path) if p<=dnS), None)
        i_slS = next((i for i,p in enumerate(path) if p>=upS), None)
        if i_tpS is not None and (i_slS is None or i_tpS<i_slS): SW+=1
        elif i_slS is not None and (i_tpS is None or i_slS<i_tpS): SL+=1
    M = max(M,1)
    pLW=LW/M; pLL=LL/M; pSW=SW/M; pSL=SL/M
    EVL = pLW*tp - pLL*sl - cost
    EVS = pSW*tp - pSL*sl - cost
    return EVL, EVS, pLW, pSW

def get_deepar_samples(symbol: str, H: int = 60, M: int = 200):
    """
    네 추론 코드에서 (M,H) 가격경로 넘겨주면 됨.
    구현 안 했으면 None 반환.
    """
    return None
def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)

def _log_trade(row:dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])

def _log_equity(eq:float):
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{eq:.6f}"])

# ===== market utils =====
LEV_CAPS: Dict[str, Tuple[float, float]] = {}

def get_symbol_leverage_caps(symbol: str) -> Tuple[float, float]:
    if symbol in LEV_CAPS:
        return LEV_CAPS[symbol]
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst = (info.get("result") or {}).get("list") or []
    if not lst:
        caps = (1.0, float(LEVERAGE))
    else:
        lf = (lst[0].get("leverageFilter") or {})
        minL = float(lf.get("minLeverage") or 1.0)
        maxL = float(lf.get("maxLeverage") or LEVERAGE)
        caps = (minL, maxL)
    LEV_CAPS[symbol] = caps
    return caps

def get_instrument_rule(symbol:str)->Dict[str,float]:
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst: return {"tickSize":0.0001,"qtyStep":0.001,"minOrderQty":0.001}
    it=lst[0]
    tick=float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
    lot =float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq=float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    return {"tickSize":tick,"qtyStep":lot,"minOrderQty":minq}

def _round_down(x:float, step:float)->float:
    if step<=0: return x
    return math.floor(x/step)*step

def _round_to_tick(x:float, tick:float)->float:
    if tick<=0: return x
    return math.floor(x/tick)*tick

def get_quote(symbol:str)->Tuple[float,float,float]:
    r=mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row=((r.get("result") or {}).get("list") or [{}])[0]
    f=lambda k: float(row.get(k) or 0.0)
    bid,ask,last=f("bid1Price"),f("ask1Price"),f("lastPrice")
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid,ask,mid or 0.0

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
            "side":p.get("side"),
            "qty":float(p.get("size") or 0.0),
            "entry":float(p.get("avgPrice") or 0.0),
            "liq":float(p.get("liqPrice") or 0.0),
            "positionIdx":int(p.get("positionIdx") or 0),
        }
    return out

def ensure_isolated_and_leverage(symbol: str):
    try:
        minL, maxL = get_symbol_leverage_caps(symbol)
        useL = max(min(float(LEVERAGE), maxL), minL)
        curL = None
        try:
            pr = client.get_positions(category=CATEGORY, symbol=symbol)
            lst = (pr.get("result") or {}).get("list") or []
            if lst:
                curL = float(lst[0].get("leverage") or 0.0)
        except Exception:
            pass
        if curL is not None and abs(curL - useL) < 1e-9:
            return
        kw = dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL))
        kw["tradeMode"] = 1  # isolated
        client.set_leverage(**kw)
        if useL < float(LEVERAGE):
            print(f"[INFO] leverage capped for {symbol}: requested {LEVERAGE} -> used {useL}")
    except Exception as e:
        print(f"[WARN] set_leverage {symbol}: {e}")

# ===== order =====
def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        return client.place_order(**params)
    except Exception as e:
        msg=str(e)
        if "position idx not match position mode" in msg and side in ("Buy","Sell"):
            try:
                p2=dict(params); p2["positionIdx"]=1 if side=="Buy" else 2
                return client.place_order(**p2)
            except Exception as e2:
                raise e2
        raise e

def place_market(symbol: str, side: str, qty: float) -> bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Market", qty=str(q), timeInForce="IOC", reduceOnly=False)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[ERR] place_market {symbol} {side} {qty} -> {e}")
        return False

def place_reduce_limit(symbol:str, side:str, qty:float, price:float)->bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit", qty=str(q), price=str(px), timeInForce="PostOnly", reduceOnly=True)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[WARN] reduce_limit {symbol} {side} {qty}@{price}: {e}")
        return False

# ===== paper 호환 사이징 =====
def compute_qty_paper_compatible(symbol: str, mid: float, equity_now: float) -> Tuple[float, float]:
    """test 코드와 동일한 notional 로직. 레버리지 캡 포함."""
    rule=get_instrument_rule(symbol)
    qty_step=float(rule["qtyStep"]); min_qty=float(rule["minOrderQty"])
    if mid<=0 or equity_now<=0: return 0.0, 0.0

    minL, maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)

    target_notional = equity_now * ENTRY_EQUITY_PCT * lv_used
    cap_abs = MAX_NOTIONAL_ABS * lv_used
    cap_pct = equity_now * lv_used * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0)
    notional_cap = min(cap_abs, cap_pct)

    notional = max(MIN_NOTIONAL_USDT, min(target_notional, notional_cap))
    qty = math.ceil((notional / mid)/qty_step)*qty_step
    if qty < min_qty:
        qty = math.ceil(min_qty/qty_step)*qty_step
    notional = qty * mid
    return qty, notional

# ===== signal =====
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
    except Exception:
        return None

def choose_entry(symbol:str)->Tuple[Optional[str], float]:
    side=None; conf=0.60
    try:
        side_cons, confs, _, _ = three_model_consensus(symbol)  # 제공되면 사용
        if side_cons in ("Buy","Sell"):
            side=side_cons
            conf=max([c for c in confs if isinstance(c,(int,float))] or [conf])
    except Exception:
        pass
    if side is None:
        side=_ma_side_from_kline(symbol)
        conf=0.60 if side else 0.0
    m=(ENTRY_MODE or "model").lower()
    if m=="inverse":
        if side=="Buy": side="Sell"
        elif side=="Sell": side="Buy"
    elif m=="random":
        side = "Buy" if (time.time()*1000)%2<1 else "Sell"
    return side, float(conf or 0.0)

# ===== stops =====
def compute_sl(entry:float, qty:float, side:str)->float:
    dyn = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    return dyn  # 페이퍼와 동일. 하드 SL 미사용

def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def set_stop_with_retry(symbol: str, sl_price: Optional[float] = None, tp_price: Optional[float] = None, position_idx: Optional[int] = None) -> bool:
    rule=get_instrument_rule(symbol); tick=float(rule.get("tickSize") or 0.0)
    if sl_price is not None: sl_price=_round_to_tick(float(sl_price), tick)
    if tp_price is not None: tp_price=_round_to_tick(float(tp_price), tick)
    params=dict(category=CATEGORY, symbol=symbol, tpslMode="Full", slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice")
    if sl_price is not None:
        params["stopLoss"]=f"{sl_price:.10f}".rstrip("0").rstrip(".")
        params["slOrderType"]="Market"
    if tp_price is not None:
        params["takeProfit"]=f"{tp_price:.10f}".rstrip("0").rstrip(".")
        params["tpOrderType"]="Market"
    if position_idx in (1,2): params["positionIdx"]=position_idx
    try:
        ret=client.set_trading_stop(**params)
        return (ret.get("retCode")==0)
    except Exception:
        return False
# --- 안전 더미: three_model_consensus 없을 때 MA fallback만 쓰도록 ---
try:
    three_model_consensus  # type: ignore[name-defined]
except NameError:
    def three_model_consensus(symbol: str):
        # (side, conf_list, extra1, extra2)
        return None, [], None, None

# --- 페이퍼/오토 공용 사이징: equity*pct*leverage, 레버리지 캡/스텝/최소수량 반영 ---
def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float):
    """
    return (qty, notional)
    - qty: 거래소 lotStep에 맞게 ceil
    - notional: qty*mid
    캡:
      MAX_NOTIONAL_ABS, MAX_NOTIONAL_PCT_OF_EQUITY, MIN_NOTIONAL_USDT
    레버리지:
      심볼별 min/max 레버리지 캡 반영
    """
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"])
    min_qty  = float(rule["minOrderQty"])
    MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "10.0"))

    if mid <= 0 or equity_now <= 0 or pct <= 0:
        return 0.0, 0.0

    # 레버리지 캡
    minL, maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)

    # 목표 노셔널(레버리지 적용)
    notional_raw = max(equity_now * pct, MIN_NOTIONAL_USDT) * lv_used
    cap_abs = MAX_NOTIONAL_ABS * lv_used
    cap_pct = equity_now * lv_used * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0)
    notional_cap = min(cap_abs, cap_pct)
    notional = max(MIN_NOTIONAL_USDT, min(notional_raw, notional_cap))

    # 수량 반올림(위쪽) 후 재계산
    qty = notional / mid
    qty = math.ceil(qty / qty_step) * qty_step
    if qty < min_qty:
        qty = math.ceil(min_qty / qty_step) * qty_step

    notional = qty * mid
    return qty, notional

# ===== entry =====
def try_enter(symbol: str, side_override: Optional[str] = None, size_pct: Optional[float] = None):
    # --- 유틸(로컬) ---
    def _force_bps(x) -> float:
        x = float(x)
        if x <= 0: return 0.0
        if x < 1.0:  return x * 100.0
        if x >= 1000.0: return 50.0
        return x
    def _force_sl_frac(x) -> float:
        x = float(x)
        if x <= 0: return 0.0
        if x >= 1.0: return min(x/100.0, 0.05)
        return x
    def _tp_px(entry: float, bps: float, side: str) -> float:
        return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

    # --- 신호 ---
    side, conf = choose_entry(symbol)
    if side_override in ("Buy","Sell"):
        side = side_override
        conf = max(conf, 0.60)
    if side not in ("Buy","Sell"):
        return

    bid, ask, mid = get_quote(symbol)
    if mid <= 0:
        return

    # --- EV 필터(옵션) ---
    if 'USE_EV' in globals() and USE_EV:
        try:
            samples = get_deepar_samples(symbol, H=60, M=200)
        except Exception:
            samples = None
        EVL, EVS, pLW, pSW = ev_long_short_from_samples(
            p0=mid,
            samples=samples,
            tp_bps=(TP1_BPS, TP2_BPS, TP3_BPS),
            sl_frac=SL_ROI_PCT,
            fee_taker=TAKER_FEE,
            slip_bps=SLIPPAGE_BPS_TAKER if EXECUTION_MODE=="taker" else SLIPPAGE_BPS_MAKER
        )
        if EVL is not None:
            if max(EVL, EVS) <= 0 and (not PAPER_COMPAT and MIN_EXPECTED_ROI_BPS > 0):
                if VERBOSE: print(f"[EV][SKIP] {symbol} EVL={EVL:.6f} EVS={EVS:.6f}")
                return
            side_ev = "Buy" if (EVL or -1) >= (EVS or -1) else "Sell"
            if not PAPER_COMPAT:
                if side_ev=="Buy" and pLW < 0.55: return
                if side_ev=="Sell" and pSW < 0.55: return
            side = side_ev

    # --- 사이징 ---
    eq = get_wallet_equity()
    use_pct = ENTRY_EQUITY_PCT if size_pct is None else float(size_pct)
    qty, _ = compute_qty_by_equity_pct(symbol, mid, eq, use_pct)
    if qty <= 0:
        return

    ensure_isolated_and_leverage(symbol)

    # --- 진입 ---
    if not place_market(symbol, side, qty):
        return

    # --- 포지션/TP·SL 세팅 ---
    pos = get_positions().get(symbol)
    if not pos:
        return
    entry = float(pos.get("entry") or mid)
    lot = get_instrument_rule(symbol)["qtyStep"]

    tp1_bps = _force_bps(TP1_BPS)
    tp2_bps = _force_bps(TP2_BPS)
    tp3_bps = _force_bps(TP3_BPS)
    sl_frac = _force_sl_frac(SL_ROI_PCT)

    tp1 = _tp_px(entry, tp1_bps, side)
    tp2 = _tp_px(entry, tp2_bps, side)
    tp3 = _tp_px(entry, tp3_bps, side)
    sl  = entry*(1.0 - sl_frac) if side=="Buy" else entry*(1.0 + sl_frac)

    qty_tp1 = _round_down(qty * float(TP1_RATIO), lot)
    qty_tp2 = _round_down(qty * float(TP2_RATIO), lot)
    close_side = "Sell" if side=="Buy" else "Buy"
    if qty_tp1 > 0: place_reduce_limit(symbol, close_side, qty_tp1, tp1)
    if qty_tp2 > 0: place_reduce_limit(symbol, close_side, qty_tp2, tp2)
    set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=pos.get("positionIdx"))

    if VERBOSE:
        print(f"[TP/SL] lev={LEVERAGE} entry={entry:.6f} tp_bps={tp1_bps}/{tp2_bps}/{tp3_bps} "
              f"tp_px={tp1:.6f}/{tp2:.6f}/{tp3:.6f} sl={sl:.6f}")

    _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":"",
                "tp1":f"{tp1:.6f}","tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}",
                "sl":f"{sl:.6f}","reason":"open","mode":ENTRY_MODE})

    # --- 예측 지평 데드라인 기록 ---
    horizon_min = int(get_signal_horizon_min(symbol)) if 'get_signal_horizon_min' in globals() else int(os.getenv("PRED_HORIZ_MIN","1"))
    if 'POSITION_DEADLINE' in globals():
        POSITION_DEADLINE[symbol] = int(time.time()) + max(1, horizon_min) * 60
        if VERBOSE:
            print(f"[DEADLINE] {symbol} +{horizon_min}m -> {POSITION_DEADLINE[symbol]}")

# ===== manage loop =====
def monitor_loop(poll_sec: float = 1.0):
    """포지션 변화 감지 + 신규 포지션 TP/SL/분할익절 세팅 + 절대익절/시간청산 + 보유시간 관리"""
    def _force_bps(x) -> float:
        x = float(x)
        if x <= 0: return 0.0
        if x < 1.0:  return x * 100.0
        if x >= 1000.0: return 50.0
        return x
    def _force_sl_frac(x) -> float:
        x = float(x)
        if x <= 0: return 0.0
        if x >= 1.0: return min(x/100.0, 0.05)
        return x
    def _tp_px(entry: float, bps: float, side: str) -> float:
        return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

    global POSITION_DEADLINE
    last_pos_keys = set()

    while True:
        try:
            _log_equity(get_wallet_equity())

            pos = get_positions()
            pos_keys = set(pos.keys())

            # 신규 포지션 세팅
            new_syms = pos_keys - last_pos_keys
            for sym in new_syms:
                p = pos[sym]
                side  = p.get("side")
                entry = float(p.get("entry") or 0.0)
                qty   = float(p.get("qty") or 0.0)
                if entry <= 0 or qty <= 0:
                    continue

                lot = get_instrument_rule(sym)["qtyStep"]

                tp1_bps = _force_bps(TP1_BPS)
                tp2_bps = _force_bps(TP2_BPS)
                tp3_bps = _force_bps(TP3_BPS)
                sl_frac = _force_sl_frac(SL_ROI_PCT)

                tp1 = _tp_px(entry, tp1_bps, side)
                tp2 = _tp_px(entry, tp2_bps, side)
                tp3 = _tp_px(entry, tp3_bps, side)
                sl  = entry*(1.0 - sl_frac) if side=="Buy" else entry*(1.0 + sl_frac)

                qty_tp1 = _round_down(qty * float(TP1_RATIO), lot)
                qty_tp2 = _round_down(qty * float(TP2_RATIO), lot)
                close_side = "Sell" if side=="Buy" else "Buy"

                if qty_tp1 > 0: place_reduce_limit(sym, close_side, qty_tp1, tp1)
                if qty_tp2 > 0: place_reduce_limit(sym, close_side, qty_tp2, tp2)
                set_stop_with_retry(sym, sl_price=sl, tp_price=tp3, position_idx=p.get("positionIdx"))

                if VERBOSE:
                    print(f"[TP/SL-SET] lev={LEVERAGE} entry={entry:.6f} "
                          f"tp_bps={tp1_bps}/{tp2_bps}/{tp3_bps} "
                          f"tp_px={tp1:.6f}/{tp2:.6f}/{tp3:.6f} sl={sl:.6f}")

                _log_trade({
                    "ts": int(time.time()), "event": "ENTRY",
                    "symbol": sym, "side": side, "qty": f"{qty:.8f}",
                    "entry": f"{entry:.6f}", "exit": "",
                    "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                    "sl": f"{sl:.6f}", "reason": "post_detect", "mode": ENTRY_MODE
                })

            # 절대익절 / 시간기반 청산
            now_ts = int(time.time())
            ABS_TP_USD = float(os.getenv("ABS_TP_USD", "1.0"))
            for sym in list(pos_keys):
                p = pos[sym]
                side = p.get("side")
                entry = float(p.get("entry") or 0.0)
                qty = float(p.get("qty") or 0.0)
                if entry <= 0 or qty <= 0:
                    continue
                _, _, mark = get_quote(sym)
                if mark <= 0:
                    continue

                pnl = unrealized_pnl_usd(side, qty, entry, mark) if 'unrealized_pnl_usd' in globals() else (
                    (mark - entry) * qty if side == "Buy" else (entry - mark) * qty)
                if pnl >= ABS_TP_USD:
                    close_side = "Sell" if side == "Buy" else "Buy"
                    _order_with_mode_retry(dict(
                        category=CATEGORY, symbol=sym, side=close_side,
                        orderType="Market", qty=str(qty),
                        reduceOnly=True, timeInForce="IOC"
                    ), side=close_side)

                    if VERBOSE: print(f"[ABS-TP] {sym} pnl={pnl:.4f} >= {ABS_TP_USD:.2f} -> CLOSE")

                    _log_trade({"ts": now_ts, "event": "EXIT", "symbol": sym, "side": side,
                                "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": f"{mark:.6f}",
                                "tp1": "", "tp2": "", "tp3": "", "sl": "",
                                "reason": "ABS_TP", "mode": ENTRY_MODE})

                    # 청산 직후 현재 소지금 기록/출력
                    cur_eq = get_wallet_equity()
                    _log_equity(cur_eq)
                    print(f"[EQUITY] {cur_eq:.2f}")

                    POSITION_DEADLINE.pop(sym, None)
                    continue

                deadline = POSITION_DEADLINE.get(sym) if 'POSITION_DEADLINE' in globals() else None
                if deadline and now_ts >= deadline:
                    close_side = "Sell" if side == "Buy" else "Buy"
                    _order_with_mode_retry(dict(
                        category=CATEGORY, symbol=sym, side=close_side,
                        orderType="Market", qty=str(qty),
                        reduceOnly=True, timeInForce="IOC"
                    ), side=close_side)

                    if VERBOSE:
                        print(f"[TIMEOUT-TP] {sym} horizon reached -> CLOSE")

                    _log_trade({"ts": now_ts, "event": "EXIT", "symbol": sym, "side": side,
                                "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": f"{mark:.6f}",
                                "tp1": "", "tp2": "", "tp3": "", "sl": "",
                                "reason": "HORIZON_TIMEOUT", "mode": ENTRY_MODE})

                    # 청산 직후 현재 소지금 기록/출력
                    cur_eq = get_wallet_equity()
                    _log_equity(cur_eq)
                    print(f"[EQUITY] {cur_eq:.2f}")

                    POSITION_DEADLINE.pop(sym, None)

            # 종료 포지션 로그
            closed_syms = last_pos_keys - pos_keys
            for sym in closed_syms:
                _, _, m = get_quote(sym)
                _log_trade({
                    "ts": int(time.time()), "event": "EXIT",
                    "symbol": sym, "side": "", "qty": "",
                    "entry": "", "exit": f"{(m or 0.0):.6f}",
                    "tp1": "", "tp2": "", "tp3": "", "sl": "",
                    "reason": "CLOSED", "mode": ENTRY_MODE
                })

            # 보유시간 초과 청산
            if MAX_HOLD_SEC > 0:
                for sym in list(pos_keys):
                    p = pos[sym]
                    et = float(p.get("updatedTime") or time.time())
                    if time.time() - et > MAX_HOLD_SEC:
                        side=p["side"]; qty=float(p["qty"])
                        close_side = "Sell" if side=="Buy" else "Buy"
                        _order_with_mode_retry(dict(
                            category=CATEGORY, symbol=sym, side=close_side,
                            orderType="Market", qty=str(qty), reduceOnly=True, timeInForce="IOC"
                        ), side=close_side)
                        _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":side,
                                    "qty":f"{qty:.8f}","entry":"","exit":"","tp1":"","tp2":"","tp3":"",
                                    "sl":"","reason":"TIMEOUT","mode":ENTRY_MODE})

            last_pos_keys = pos_keys
            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("[MONITOR] stop"); break
        except Exception as e:
            print("[MONITOR ERR]", e); time.sleep(1.0)

# ===== example main =====
def main():
    # 라운드 로빈으로 기회 탐색. 간단화.
    while True:
        for s in SYMBOLS:
            try_enter(s)
        # 모니터 1~2초만 돌고 빠져나오기
        for _ in range(2):
            monitor_loop(1.0)  # monitor_loop를 '한 번 돌고 리턴' 형태로 리팩터 필요

if __name__ == "__main__":
    main()
