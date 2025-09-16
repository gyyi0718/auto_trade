# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — PAPER 호환 + HORIZON_TIMEOUT/ABS_TP
- 페이퍼 코드의 [ABS-TP], [TIMEOUT-TP] 동작을 동일하게 구현
- 데드라인 도달 시 시장가 청산 + 로그 기록(reason=HORIZON_TIMEOUT)
- 절대익절 달성 시 시장가 청산 + 로그 기록(reason=ABS_TP)
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
LEVERAGE   = float(os.getenv("LEVERAGE", "10"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "taker").lower()
EXIT_EXEC_MODE  = os.getenv("EXIT_EXEC_MODE", "taker").lower()
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")
MAX_OPEN   = int(os.getenv("MAX_OPEN", "5"))

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
FIXED_TP_SL = str(os.getenv("FIXED_TP_SL", "1")).lower() in ("1","y","yes","true")

# === 페이퍼 호환 강제 ===
if PAPER_COMPAT:
    EXECUTION_MODE = "taker"
    EXIT_EXEC_MODE = "taker"
    SLIPPAGE_BPS_TAKER = 1.0
    SLIPPAGE_BPS_MAKER = 0.0
    MAKER_FEE = 0.0
    MIN_EXPECTED_ROI_BPS = -1.0

# ===== 절대익절/타임아웃(페이퍼와 동일 동작) =====
ABS_TP_USD = float(os.getenv("ABS_TP_USD", "1.0"))        # 수수료 제외 달러 기준 이익
PRED_HORIZ_MIN_DEFAULT = int(os.getenv("PRED_HORIZ_MIN", "1"))

# ===== I/O =====
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
        csv.writer(f).writerow([int(time.time()), f"{float(eq):.6f}"])
    print(f"[EQUITY] {float(eq):.2f}")

# ===== API =====
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = ""
API_SECRET = ""
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

SYMBOLS = os.getenv("SYMBOLS","BTCUSDT,ETHUSDT,SOLUSDT").split(",")

# ===== state =====
POSITION_DEADLINE: Dict[str,int] = {}   # {symbol: epoch_seconds}
LEV_CAPS: Dict[str, Tuple[float, float]] = {}

# ===== market utils =====
def get_symbol_leverage_caps(symbol: str) -> Tuple[float, float]:
    if symbol in LEV_CAPS: return LEV_CAPS[symbol]
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst = (info.get("result") or {}).get("list") or []
    if not lst:
        caps = (1.0, float(LEVERAGE))
    else:
        lf = (lst[0].get("leverageFilter") or {})
        caps = (float(lf.get("minLeverage") or 1.0), float(lf.get("maxLeverage") or LEVERAGE))
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
        kw = dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL))
        kw["tradeMode"] = 1  # isolated
        client.set_leverage(**kw)
        if useL < float(LEVERAGE):
            print(f"[INFO] leverage capped for {symbol}: requested {LEVERAGE} -> used {useL}")
    except Exception as e:
        print(f"[WARN] set_leverage {symbol}: {e}")

# ===== orders =====
def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        return client.place_order(**params)
    except Exception as e:
        msg=str(e)
        if "position idx not match position mode" in msg and side in ("Buy","Sell"):
            p2=dict(params); p2["positionIdx"]=1 if side=="Buy" else 2
            return client.place_order(**p2)
        raise

def place_market(symbol: str, side: str, qty: float, reduce_only: bool=False) -> bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"])
        if q<rule["minOrderQty"]: return False
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Market",
               qty=str(q), timeInForce="IOC", reduceOnly=reduce_only)
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
        p=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit", qty=str(q),
               price=str(px), timeInForce="PostOnly", reduceOnly=True)
        _order_with_mode_retry(p, side=side)
        return True
    except Exception as e:
        print(f"[WARN] reduce_limit {symbol} {side} {qty}@{price}: {e}")
        return False

# ===== sizing =====
def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float):
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"])
    min_qty  = float(rule["minOrderQty"])
    if mid <= 0 or equity_now <= 0 or pct <= 0:
        return 0.0, 0.0
    minL, maxL = get_symbol_leverage_caps(symbol)
    lv_used = max(min(float(LEVERAGE), maxL), minL)
    notional_raw = max(equity_now * pct, MIN_NOTIONAL_USDT) * lv_used
    cap_abs = MAX_NOTIONAL_ABS * lv_used
    cap_pct = equity_now * lv_used * (MAX_NOTIONAL_PCT_OF_EQUITY / 100.0)
    notional_cap = min(cap_abs, cap_pct)
    notional = max(MIN_NOTIONAL_USDT, min(notional_raw, notional_cap))
    qty = notional / mid
    qty = math.ceil(qty / qty_step) * qty_step
    if qty < min_qty:
        qty = math.ceil(min_qty / qty_step) * qty_step
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

def three_model_consensus(symbol: str):
    return None, [], None, None

def choose_entry(symbol:str)->Tuple[Optional[str], float]:
    side=None; conf=0.60
    side_cons, confs, _, _ = three_model_consensus(symbol)
    if side_cons in ("Buy","Sell"):
        side=side_cons
        if confs: conf=max([c for c in confs if isinstance(c,(int,float))] or [conf])
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

def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def get_signal_horizon_min(symbol: str) -> int:
    return PRED_HORIZ_MIN_DEFAULT

# ===== PnL helpers (페이퍼와 동일 포맷 출력) =====
def unrealized_pnl_usd(side: str, qty: float, entry: float, mark: float) -> float:
    return (mark - entry) * qty if side=="Buy" else (entry - mark) * qty

def apply_exec_pnl_and_log_equity(side: str, entry: float, exit_px: float, qty: float):
    gross = unrealized_pnl_usd(side, qty, entry, exit_px)
    fee = (entry*qty + exit_px*qty) * TAKER_FEE  # taker 가정
    eq = get_wallet_equity()  # 실거래는 지갑이 소스
    _log_equity(eq)          # 현재 지갑 equity를 기록(실현 후 반영 가정)
    return gross, fee, eq

# ===== entry/exit =====
def try_enter(symbol: str, side_override: Optional[str] = None, size_pct: Optional[float] = None):
    # 신호
    side, conf = choose_entry(symbol)
    if side_override in ("Buy","Sell"):
        side = side_override
        conf = max(conf, 0.60)
    if side not in ("Buy","Sell"):
        return

    # 견적
    _, _, mid = get_quote(symbol)
    if mid <= 0:
        return

    # 사이징
    eq = get_wallet_equity()
    use_pct = ENTRY_EQUITY_PCT if size_pct is None else float(size_pct)
    qty, _ = compute_qty_by_equity_pct(symbol, mid, eq, use_pct)
    if qty <= 0:
        return

    ensure_isolated_and_leverage(symbol)
    if not place_market(symbol, side, qty, reduce_only=False):
        return

    # 포지션 확인
    pos = get_positions().get(symbol)
    if not pos:
        return
    entry = float(pos.get("entry") or mid)
    lot = get_instrument_rule(symbol)["qtyStep"]

    # TP/SL 표시용
    tp1 = tp_from_bps(entry, TP1_BPS, side)
    tp2 = tp_from_bps(entry, TP2_BPS, side)
    tp3 = tp_from_bps(entry, TP3_BPS, side)
    sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)

    qty_tp1 = _round_down(qty * float(TP1_RATIO), lot)
    qty_tp2 = _round_down(qty * float(TP2_RATIO), lot)
    close_side = "Sell" if side=="Buy" else "Buy"
    if qty_tp1 > 0: place_reduce_limit(symbol, close_side, qty_tp1, tp1)
    if qty_tp2 > 0: place_reduce_limit(symbol, close_side, qty_tp2, tp2)
    # 주: set_trading_stop 사용 가능하지만 페이퍼 동작 일치 목적이면 생략 가능

    # 엔트리 로그
    _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":symbol,"side":side,
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":"",
                "tp1":f"{tp1:.6f}","tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}",
                "sl":f"{sl:.6f}","reason":"open","mode":ENTRY_MODE})

    # 데드라인
    horizon_min = int(get_signal_horizon_min(symbol))
    POSITION_DEADLINE[symbol] = int(time.time()) + max(1, horizon_min) * 60
    print(f"[DEADLINE] {symbol} +{horizon_min}m -> {POSITION_DEADLINE[symbol]}")

def close_market(symbol: str):
    pos = get_positions().get(symbol)
    if not pos: return None
    side = pos["side"]; qty = float(pos["qty"]); entry = float(pos["entry"])
    close_side = "Sell" if side=="Buy" else "Buy"
    ok = place_market(symbol, close_side, qty, reduce_only=True)
    if not ok: return None
    # 체결 후 현재가로 gross/fee 계산
    _, _, mark = get_quote(symbol)
    gross, fee, new_eq = apply_exec_pnl_and_log_equity(side, entry, mark, qty)
    _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":symbol,"side":close_side,
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":"timeout-close","mode":ENTRY_MODE})
    return gross, fee, new_eq, mark

# ===== monitor loop (페이퍼와 동일한 ABS_TP/타임아웃 처리) =====
def monitor_loop(poll_sec: float = 1.0, max_loops: int = 2):
    loops = 0
    while loops < max_loops:
        try:
            # 지갑 equity 스냅샷 기록
            _log_equity(get_wallet_equity())
            now = int(time.time())

            # 포지션 스캔
            pos = get_positions()
            for sym, p in list(pos.items()):
                side = p["side"]; entry = float(p["entry"]); qty = float(p["qty"])
                _, _, mark = get_quote(sym)
                if mark <= 0:
                    continue
                pnl = unrealized_pnl_usd(side, qty, entry, mark)

                # 1) 절대익절
                if pnl >= ABS_TP_USD:
                    ret = close_market(sym)
                    if ret:
                        gross, fee, new_eq, ex = ret
                        print(f"[ABS-TP] {sym} gross={gross:.6f} fee={fee:.6f} eq={new_eq:.2f}")
                        _log_trade({"ts": now, "event": "EXIT", "symbol": sym, "side": ("Sell" if side=="Buy" else "Buy"),
                                    "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": f"{ex:.6f}",
                                    "tp1": "", "tp2": "", "tp3": "", "sl": "",
                                    "reason": "ABS_TP", "mode": ENTRY_MODE})
                        POSITION_DEADLINE.pop(sym, None)
                        continue

                # 2) 시간기반 청산
                deadline = POSITION_DEADLINE.get(sym)
                if deadline and now >= deadline:
                    ret = close_market(sym)
                    if ret:
                        gross, fee, new_eq, ex = ret
                        print(f"[TIMEOUT-TP] {sym} gross={gross:.6f} fee={fee:.6f} [EQUITY]={new_eq:.2f}")
                        _log_trade({"ts": now, "event": "EXIT", "symbol": sym, "side": ("Sell" if side=="Buy" else "Buy"),
                                    "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": f"{ex:.6f}",
                                    "tp1": "", "tp2": "", "tp3": "", "sl": "",
                                    "reason": "HORIZON_TIMEOUT", "mode": ENTRY_MODE})
                        POSITION_DEADLINE.pop(sym, None)

            loops += 1
            time.sleep(poll_sec)
        except Exception as e:
            print("[MONITOR ERR]", e)
            time.sleep(1.0)
            loops += 1

# ===== main =====
def main():
    while True:
        # 라운드로빈 진입 시도
        for s in SYMBOLS:
            try:
                try_enter(s)
            except Exception as e:
                print("[ENTER ERR]", s, e)
        # 짧게 모니터링(ABS_TP/타임아웃 처리)
        monitor_loop(poll_sec=1.0, max_loops=2)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
