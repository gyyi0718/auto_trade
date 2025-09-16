# -*- coding: utf-8 -*-
"""
Binance Futures (USDT-Perp) Live Trading — unified, maker/taker, TP1/TP2 reduceOnly, TP3/SL stop
- Bybit용 스크립트를 Binance로 포팅. 구조/동작 동일하게 유지.
- ENTRY_MODE=model|inverse|random, 최소 기대 ROI 필터, PostOnly 대기·취소, BE/트레일, 시간청산, 글로벌 쿨다운.

ENV (예시)
  BINANCE_API_KEY=...
  BINANCE_API_SECRET=...
  BINANCE_TESTNET=0                # 1이면 테스트넷
  ENTRY_MODE=model                 # model|inverse|random
  LEVERAGE=10
  RISK_PCT_OF_EQUITY=1.0
  ENTRY_PORTION=0.75
  KELLY_MAX_ON_CASH=0.5
  MIN_NOTIONAL=25.0
  MAX_NOTIONAL_ABS=2000.0
  MAX_NOTIONAL_PCT_OF_EQUITY=20.0
  MIN_NOTIONAL_USDT=10.0

  EXECUTION_MODE=maker             # maker|taker
  EXIT_EXEC_MODE=taker             # maker|taker
  TAKER_FEE=0.0004                 # 계정 수수료 반영
  MAKER_FEE=0.0002
  SLIPPAGE_BPS_TAKER=1.0
  SLIPPAGE_BPS_MAKER=0.0
  MIN_EXPECTED_ROI_BPS=20.0
  WAIT_FOR_MAKER_FILL_SEC=30
  CANCEL_UNFILLED_MAKER=1

  SL_ROI_PCT=0.01                  # 동적 SL 1%
  MAX_HOLD_SEC=3600
  COOLDOWN_SEC=60
  NEG_LOSS_GLOBAL_COOLDOWN_SEC=180
  MAX_OPEN=10

실행
  python binance_autotrade.py
"""

import os, time, math, csv, threading
from typing import Optional, Dict, Tuple, List

try:
    # pip install binance-connector
    from binance.um_futures import UMFutures
except Exception as e:
    raise RuntimeError("pip install binance-connector") from e

# ===== env =====
CATEGORY = "linear"  # 호환성 표기만 유지
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")
LEVERAGE   = float(os.getenv("LEVERAGE", "10"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0004"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0002"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "maker").lower()
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower()
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")
MAX_OPEN   = int(os.getenv("MAX_OPEN", "10"))

SL_ROI_PCT      = float(os.getenv("SL_ROI_PCT", "0.01"))
HARD_SL_PCT     = 0.01
TP1_BPS,TP2_BPS,TP3_BPS = 35.0, 100.0, 200.0
TP1_RATIO,TP2_RATIO     = 0.40, 0.35
BE_EPS_BPS = 2.0
TRAIL_BPS = 50.0
TRAIL_AFTER_TIER = 2

SYMBOLS=[
  "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
  "LINKUSDT","ADAUSDT","SUIUSDT","1000PEPEUSDT","MNTUSDT",
  "AVAXUSDT","APTUSDT","BNBUSDT","UNIUSDT","LTCUSDT"
]

RISK_PCT_OF_EQUITY = float(os.getenv("RISK_PCT_OF_EQUITY","1.0"))
ENTRY_PORTION = float(os.getenv("ENTRY_PORTION","0.75"))
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL","25.0"))
MAX_NOTIONAL_ABS = float(os.getenv("MAX_NOTIONAL_ABS","2000.0"))
MAX_NOTIONAL_PCT_OF_EQUITY = float(os.getenv("MAX_NOTIONAL_PCT_OF_EQUITY","20.0"))
KELLY_MAX_ON_CASH = float(os.getenv("KELLY_MAX_ON_CASH","0.5"))

MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","60"))
NEG_LOSS_GLOBAL_COOLDOWN_SEC=int(os.getenv("NEG_LOSS_GLOBAL_COOLDOWN_SEC","180"))
GLOBAL_COOLDOWN_UNTIL = 0.0

TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]

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

# ===== clients =====
TESTNET = str(os.getenv("BINANCE_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = os.getenv("BINANCE_API_KEY","")
API_SECRET = os.getenv("BINANCE_API_SECRET","")

BASE_URL = "https://testnet.binancefuture.com" if TESTNET else "https://fapi.binance.com"
client = UMFutures(key=API_KEY, secret=API_SECRET, base_url=BASE_URL)
mkt    = UMFutures(base_url=BASE_URL)  # 공개 API

# ===== exchange meta =====
RULES: Dict[str,dict] = {}
LEV_CAPS: Dict[str, Tuple[float,float]] = {}

def _fetch_exchange_info():
    info = mkt.exchange_info()
    for s in info.get("symbols", []):
        sym = s.get("symbol")
        if not sym: continue
        tick=qty_step= min_qty = 0.0
        minL, maxL = 1.0, float(LEVERAGE)
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick = float(f.get("tickSize", "0.0001"))
            if f.get("filterType") == "LOT_SIZE":
                qty_step = float(f.get("stepSize", "0.001"))
                min_qty  = float(f.get("minQty", "0.001"))
            if f.get("filterType") == "LEVERAGE":
                minL = float(f.get("minQty", "1"))  # Binance는 별도 레버리지 필터가 아님. 안전상 기본값 유지
        RULES[sym] = {"tickSize":tick or 0.0001, "qtyStep":qty_step or 0.001, "minOrderQty":max(min_qty, qty_step or 0.001)}
        LEV_CAPS[sym] = (1.0, float(LEVERAGE))

_fetch_exchange_info()

def get_instrument_rule(symbol:str)->dict:
    return RULES.get(symbol) or {"tickSize":0.0001,"qtyStep":0.001,"minOrderQty":0.001}

def get_symbol_leverage_caps(symbol:str)->Tuple[float,float]:
    return LEV_CAPS.get(symbol) or (1.0, float(LEVERAGE))

def _round_down(x:float, step:float)->float:
    if step<=0: return x
    return math.floor(x/step)*step

def _round_to_tick(x:float, tick:float)->float:
    if tick<=0: return x
    return math.floor(x/tick)*tick

# ===== market =====
def get_quote(symbol:str)->Tuple[float,float,float]:
    bt = mkt.book_ticker(symbol=symbol) or {}
    bid = float(bt.get("bidPrice") or 0.0)
    ask = float(bt.get("askPrice") or 0.0)
    if bid>0 and ask>0:
        mid = (bid+ask)/2.0
    else:
        pi = mkt.premium_index(symbol=symbol) or {}
        last = float(pi.get("markPrice") or 0.0)
        mid = last or bid or ask
    return bid, ask, mid

def _ma_side_from_kline(symbol:str, interval:str="1m", limit:int=120)->Optional[str]:
    try:
        rows = mkt.klines(symbol=symbol, interval=interval, limit=min(int(limit),1000)) or []
        closes = [float(r[4]) for r in rows]
        if len(closes) < 60: return None
        ma20 = sum(closes[-20:])/20.0
        ma60 = sum(closes[-60:])/60.0
        if ma20>ma60: return "Buy"
        if ma20<ma60: return "Sell"
        return None
    except Exception:
        return None

# ===== wallet/positions =====
def get_wallet_equity()->float:
    try:
        bal = client.balance() or []
        total=0.0
        for c in bal:
            if c.get("asset")=="USDT":
                total += float(c.get("balance") or 0.0)
                total += float(c.get("crossUnPnl") or 0.0)
        return float(total)
    except Exception:
        return 0.0

def get_positions()->Dict[str,dict]:
    out={}
    try:
        lst = client.position_information() or []
        for p in lst:
            sym=p.get("symbol"); qty=float(p.get("positionAmt") or 0.0)
            if not sym or abs(qty)<=0: continue
            side = "Buy" if qty>0 else "Sell"
            out[sym]={
                "side":side,
                "qty":abs(qty),
                "entry":float(p.get("entryPrice") or 0.0),
                "liq":float(p.get("liquidationPrice") or 0.0),
            }
    except Exception:
        pass
    return out

# ===== leverage / margin / orders =====
def ensure_isolated_and_leverage(symbol:str):
    try:
        # 마진 모드
        try:
            client.change_margin_type(symbol=symbol, marginType="ISOLATED")
        except Exception:
            pass
        # 레버리지
        minL,maxL = get_symbol_leverage_caps(symbol)
        useL = max(min(float(LEVERAGE), maxL), minL)
        try:
            client.change_leverage(symbol=symbol, leverage=int(useL))
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] leverage/margin {symbol}: {e}")

def _order(params:dict):
    return client.new_order(**params)

def place_market(symbol:str, side:str, qty:float)->bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        if q<rule["minOrderQty"]: return False
        _order(dict(symbol=symbol, side=side.upper(), type="MARKET", quantity=f"{q}"))
        return True
    except Exception as e:
        print(f"[ERR] market {symbol} {side} {qty}: {e}")
        return False

def place_limit_postonly(symbol:str, side:str, qty:float, price:float)->Optional[int]:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return None
        ret=_order(dict(symbol=symbol, side=side.upper(), type="LIMIT", timeInForce="GTX",  # PostOnly=GTX
                        quantity=f"{q}", price=f"{px}"))
        oid = ret.get("orderId")
        print(f"[ORDER][POST_ONLY] {symbol} {side} {q}@{px} oid={oid}")
        return oid
    except Exception as e:
        print(f"[ERR] postonly {symbol} {side} {qty}@{price}: {e}")
        return None

def place_reduce_limit(symbol:str, side:str, qty:float, price:float)->bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        px=_round_to_tick(price, rule["tickSize"])
        if q<rule["minOrderQty"]: return False
        _order(dict(symbol=symbol, side=side.upper(), type="LIMIT", timeInForce="GTC",
                    quantity=f"{q}", price=f"{px}", reduceOnly="true"))
        return True
    except Exception as e:
        print(f"[WARN] reduce_limit {symbol} {side} {qty}@{price}: {e}")
        return False

def set_stop_with_retry(symbol:str, sl_price:Optional[float]=None, tp_price:Optional[float]=None)->bool:
    ok=True
    try:
        if sl_price is not None:
            client.new_order(symbol=symbol, side="SELL", type="STOP_MARKET", stopPrice=f"{sl_price}",
                             closePosition="true", workingType="MARK_PRICE")
            client.new_order(symbol=symbol, side="BUY",  type="STOP_MARKET", stopPrice=f"{sl_price}",
                             closePosition="true", workingType="MARK_PRICE")
        if tp_price is not None:
            client.new_order(symbol=symbol, side="SELL", type="TAKE_PROFIT_MARKET", stopPrice=f"{tp_price}",
                             closePosition="true", workingType="MARK_PRICE")
            client.new_order(symbol=symbol, side="BUY",  type="TAKE_PROFIT_MARKET", stopPrice=f"{tp_price}",
                             closePosition="true", workingType="MARK_PRICE")
    except Exception as e:
        print(f"[STOP ERR] {symbol}: {e}")
        ok=False
    return ok

# ===== calcs =====
def compute_sl(entry:float, qty:float, side:str)->float:
    dyn = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    hard = entry*(1.0 - HARD_SL_PCT) if side=="Buy" else entry*(1.0 + HARD_SL_PCT)
    return max(dyn, hard) if side=="Buy" else min(dyn, hard)

def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _bps_between(p1:float, p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps()->float:
    fee = (0.0 if EXECUTION_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXECUTION_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))

def _exit_cost_bps()->float:
    fee = (0.0 if EXIT_EXEC_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXIT_EXEC_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))

def compute_position_size(symbol:str, mid:float, equity_now:float, free_cash:float, conf:float)->Tuple[float,float]:
    rule=get_instrument_rule(symbol)
    qty_step=float(rule["qtyStep"]); min_qty=float(rule["minOrderQty"])
    MIN_NOTIONAL_USDT=float(os.getenv("MIN_NOTIONAL_USDT","10.0"))

    use_cash_cap = equity_now * RISK_PCT_OF_EQUITY
    use_cash     = min(free_cash * ENTRY_PORTION, use_cash_cap)
    if use_cash<=0 or mid<=0: return 0.0, 0.0

    kelly_like = max(0.05, min(conf, KELLY_MAX_ON_CASH))
    use_final_cash = use_cash * kelly_like

    minL,maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)
    notional_raw = use_final_cash * lv_used
    notional_cap = min(MAX_NOTIONAL_ABS, equity_now * (MAX_NOTIONAL_PCT_OF_EQUITY/100.0))
    notional = max(MIN_NOTIONAL, min(notional_raw, notional_cap))
    notional = max(notional, MIN_NOTIONAL_USDT)

    qty = notional / mid
    qty = math.ceil(qty/qty_step)*qty_step
    if qty < min_qty: qty = math.ceil(min_qty/qty_step)*qty_step
    notional = qty * mid
    return qty, notional

# ===== signal glue =====
def three_model_consensus(symbol:str):
    # 자리표시자: 외부 모델을 쓰는 경우 여기에 연결
    # return side_cons, [conf1,conf2,conf3], details1, details2
    raise RuntimeError("three_model_consensus not wired")

def choose_entry(symbol:str)->Tuple[Optional[str], float]:
    side=None; conf=0.60
    try:
        side_cons, confs, _, _ = three_model_consensus(symbol)  # 외부 구현 시 사용
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

# ===== states =====
STATE: Dict[str,dict] = {}
COOLDOWN_UNTIL: Dict[str,float] = {}
PENDING_MAKER: Dict[str,dict] = {}  # symbol -> {orderId, ts, side, qty, price}

def _init_state(symbol:str, side:str, entry:float, qty:float, lot_step:float):
    STATE[symbol] = {
        "side":side, "entry":float(entry), "init_qty":float(qty),
        "tp_done":[False,False,False], "be_moved":False,
        "entry_ts":time.time(), "lot_step":float(lot_step),
    }

# ===== entry/monitor =====
def try_enter(symbol:str):
    global GLOBAL_COOLDOWN_UNTIL
    if time.time() < GLOBAL_COOLDOWN_UNTIL: return
    if len(get_positions()) >= MAX_OPEN: return
    if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return
    if symbol in PENDING_MAKER: return

    side, conf = choose_entry(symbol)
    if side not in ("Buy","Sell") or conf < 0.55: return

    bid,ask,mid = get_quote(symbol)
    if mid <= 0: return

    rule=get_instrument_rule(symbol)
    entry_ref = float(ask if side=="Buy" else bid)
    tp1_px = _round_to_tick(tp_from_bps(entry_ref, TP1_BPS, side), rule["tickSize"])
    edge_bps = _bps_between(entry_ref, tp1_px)
    cost_bps = _entry_cost_bps() + _exit_cost_bps()
    need_bps = cost_bps + MIN_EXPECTED_ROI_BPS
    if edge_bps < need_bps:
        return

    eq = get_wallet_equity()
    free_cash = eq
    qty, notional = compute_position_size(symbol, mid, eq, free_cash, conf)
    if qty <= 0: return

    ensure_isolated_and_leverage(symbol)

    if EXECUTION_MODE == "maker":
        limit_px = bid if side=="Buy" else ask
        oid = place_limit_postonly(symbol, side, qty, limit_px)
        if oid is not None:
            PENDING_MAKER[symbol] = {"orderId": oid, "ts": time.time(), "side": side, "qty": qty, "price": limit_px}
            _log_trade({"ts":int(time.time()),"event":"ENTRY_POSTONLY","symbol":symbol,"side":side,
                        "qty":f"{qty:.8f}","entry":f"{limit_px:.6f}","exit":"","tp1":f"{tp1_px:.6f}",
                        "tp2":"","tp3":"","sl":"","reason":"open","mode":ENTRY_MODE})
    else:
        ok = place_market(symbol, side, qty)
        if not ok: return
        # position refresh
        pos = get_positions().get(symbol)
        if not pos: return
        entry = float(pos.get("entry") or mid)
        lot = get_instrument_rule(symbol)["qtyStep"]
        sl = compute_sl(entry, qty, side)
        tp1 = tp_from_bps(entry, TP1_BPS, side)
        tp2 = tp_from_bps(entry, TP2_BPS, side)
        tp3 = tp_from_bps(entry, TP3_BPS, side)
        qty_tp1 = _round_down(qty * TP1_RATIO, lot)
        qty_tp2 = _round_down(qty * TP2_RATIO, lot)
        close_side = "SELL" if side=="Buy" else "BUY"
        if qty_tp1>0: place_reduce_limit(symbol, close_side, qty_tp1, tp1)
        if qty_tp2>0: place_reduce_limit(symbol, close_side, qty_tp2, tp2)
        set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3)
        _init_state(symbol, side, entry, qty, lot)
        _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                    "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":"","tp1":f"{tp1:.6f}",
                    "tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}","sl":f"{sl:.6f}","reason":"open","mode":ENTRY_MODE})

def _update_trailing_and_be(symbol:str):
    st = STATE.get(symbol)
    if not st: return
    pos = get_positions().get(symbol)
    if not pos: return
    side=st["side"]; entry=st["entry"]
    qty=float(pos.get("qty") or 0.0)
    if qty<=0: return

    _,_,mark = get_quote(symbol)

    init_qty = float(st.get("init_qty",0.0))
    rem = qty / max(init_qty, 1e-12)
    if not st["tp_done"][0] and rem <= (1.0 - TP1_RATIO + 1e-8):
        st["tp_done"][0]=True; _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":ENTRY_MODE})
    if not st["tp_done"][1] and rem <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1]=True; _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":ENTRY_MODE})

    be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
    if st.get("tp_done",[False,False,False])[1] and not st.get("be_moved",False):
        set_stop_with_retry(symbol, sl_price=be_px, tp_price=None)
        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":symbol,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})

    if st.get("tp_done",[False,False,False])[TRAIL_AFTER_TIER-1]:
        if side=="Buy":
            new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
        else:
            new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=new_sl, tp_price=None)

def _check_time_stop_and_cooldowns(symbol:str):
    st=STATE.get(symbol); pos=get_positions().get(symbol)
    if not st or not pos: return
    if MAX_HOLD_SEC>0 and (time.time()-st.get("entry_ts",time.time()))>MAX_HOLD_SEC:
        side=pos["side"]; qty=float(pos["qty"])
        close_side = "SELL" if side=="Buy" else "BUY"
        try:
            _order(dict(symbol=symbol, side=close_side, type="MARKET", quantity=f"{qty}", reduceOnly="true"))
        except Exception as e:
            print("[TIMEOUT CLOSE ERR]", e)
        COOLDOWN_UNTIL[symbol]=time.time()+COOLDOWN_SEC
        _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":symbol,"side":side,"qty":f"{qty:.8f}","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"TIMEOUT","mode":ENTRY_MODE})

# ===== pending postonly =====
def _get_open_orders(symbol: Optional[str]=None)->List[dict]:
    try:
        return client.get_open_orders(symbol=symbol) or []
    except Exception:
        return []

def _cancel_order(symbol:str, order_id:int)->bool:
    try:
        client.cancel_order(symbol=symbol, orderId=order_id); return True
    except Exception as e:
        print(f"[CANCEL ERR] {symbol} {order_id}: {e}"); return False

def process_pending_postonly():
    now=time.time()
    if not PENDING_MAKER: return
    for sym in list(PENDING_MAKER.keys()):
        oinfo=PENDING_MAKER.get(sym) or {}
        oid=oinfo.get("orderId"); side=oinfo.get("side")
        open_list=_get_open_orders(sym)
        still_open = any((x.get("orderId")==oid) for x in open_list)
        pos_now = get_positions().get(sym)

        if not still_open:
            PENDING_MAKER.pop(sym, None)
            # 체결되면 monitor 루프에서 후속 처리
            continue

        if CANCEL_UNFILLED_MAKER and (now - float(oinfo.get("ts", now)) > WAIT_FOR_MAKER_FILL_SEC):
            ok=_cancel_order(sym, oid)
            PENDING_MAKER.pop(sym, None)
            if ok:
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts":int(time.time()),"event":"CANCEL_POSTONLY","symbol":sym,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"TIMEOUT","mode":ENTRY_MODE})

# ===== loops =====
def monitor_loop(poll_sec:float=1.0):
    global GLOBAL_COOLDOWN_UNTIL
    last_pos_keys=set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            process_pending_postonly()

            pos=get_positions(); pos_keys=set(pos.keys())
            for sym in (last_pos_keys - pos_keys):
                STATE.pop(sym, None)
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":"","qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"CLOSED","mode":ENTRY_MODE})

            for sym in pos_keys:
                _update_trailing_and_be(sym)
                _check_time_stop_and_cooldowns(sym)

            last_pos_keys=pos_keys
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[MONITOR ERR]",e); time.sleep(2.0)

def _top_symbols_by_volume(whitelist:List[str], top_n:int=MAX_OPEN)->List[str]:
    # 단순화: 화이트리스트 그대로 반환(필요시 24h 통계로 정렬 가능)
    return whitelist[:top_n]

def scanner_loop(iter_delay:float=2.5):
    while True:
        try:
            if time.time()<GLOBAL_COOLDOWN_UNTIL: time.sleep(iter_delay); continue
            syms=_top_symbols_by_volume(SYMBOLS, top_n=min(MAX_OPEN,len(SYMBOLS)))
            if len(get_positions())>=MAX_OPEN: time.sleep(iter_delay); continue
            for s in syms:
                if s in PENDING_MAKER: continue
                if len(get_positions())>=MAX_OPEN: break
                try_enter(s)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[SCANNER ERR]",e); time.sleep(2.0)

def main():
    print(f"[START] Binance Futures MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} TESTNET={TESTNET}")
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2=threading.Thread(target=scanner_loop, args=(2.5,), daemon=True)
    t1.start(); t2.start()
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
