# -*- coding: utf-8 -*-
import os, time, csv, math, threading, traceback, re
from typing import Dict, Optional

CATEGORY = "linear"
MIRROR_SOURCE_CSV = os.getenv("MIRROR_SOURCE_CSV","paper_trades_model_1000.0_20.0.csv")
MIRROR_TARGET = os.getenv("MIRROR_TARGET","paper").lower()  # live|paper
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
START_EQUITY = float(os.getenv("START_EQUITY","1000"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
EQUITY_PRINT_SEC = int(os.getenv("EQUITY_PRINT_SEC","10"))
VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

src = MIRROR_SOURCE_CSV
EQUITY_CSV_RENAME = re.sub(r'(^|_)trades_', r'\1equity_', src, count=1)
TRADES_CSV = os.getenv("TRADES_CSV", MIRROR_SOURCE_CSV.replace("model","inverse"))
EQUITY_CSV = os.getenv("EQUITY_CSV", EQUITY_CSV_RENAME)

TRADES_FIELDS = ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS = ["ts","equity","target"]

API_KEY = os.getenv("BYBIT_API_KEY","")
API_SECRET = os.getenv("BYBIT_API_SECRET","")

client = None; mkt = None
if MIRROR_TARGET == "live":
    from pybit.unified_trading import HTTP
    client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
    mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

def _ensure_csvs():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(EQUITY_FIELDS)

def _log_trade(row: dict):
    _ensure_csvs()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])

def _log_equity(eq: float):
    _ensure_csvs()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{eq:.6f}", MIRROR_TARGET])
    if VERBOSE: print(f"[EQUITY][{MIRROR_TARGET}] {eq:.2f}")

_RULE_CACHE: Dict[str,dict] = {}
def get_rule(symbol: str) -> Dict[str,float]:
    if MIRROR_TARGET == "paper":
        return {"qtyStep": 0.001, "minOrderQty": 0.001}
    v = _RULE_CACHE.get(symbol)
    if v: return v
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    it = ((info.get("result") or {}).get("list") or [{}])[0]
    lot = float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq= float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    _RULE_CACHE[symbol] = {"qtyStep":lot, "minOrderQty":minq}
    return _RULE_CACHE[symbol]

def _fmt_qty(q: float, step: float) -> str:
    s = f"{step:.16f}".rstrip("0"); dec = 0 if "." not in s else len(s.split(".")[1])
    return (f"{q:.{dec}f}" if dec>0 else str(int(round(q)))).rstrip("0").rstrip(".")

def place_market_live(symbol: str, side: str, qty: float) -> Optional[float]:
    rule = get_rule(symbol); step = rule["qtyStep"]; minq = rule["minOrderQty"]
    q = math.floor(float(qty)/step)*step
    if q < minq:
        if VERBOSE: print(f"[SKIP] {symbol} q<{minq}")
        return None
    r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0); ask=float(row.get("ask1Price") or 0.0); last=float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    tries = 0
    while tries<4 and q>=minq:
        try:
            client.place_order(category=CATEGORY, symbol=symbol, side=side,
                               orderType="Market", qty=_fmt_qty(q, step),
                               timeInForce="IOC", reduceOnly=(side=="Buy" or side=="Sell") and False)
            return float(mid or 0.0)
        except Exception as e:
            msg=str(e)
            if "110007" in msg or "not enough" in msg:
                q = math.floor((q*0.7)/step)*step; tries+=1; continue
            time.sleep(0.3); tries+=1
    return None

# 포지션 관리: 역포지션을 실제로 열고/닫아 PnL 계산
# pos[sym] = {"side":"Buy|Sell", "qty":float, "avg":float}
pos: Dict[str,Dict[str,float]] = {}
_paper_eq = START_EQUITY

def _open_inverse(sym: str, model_side: str, qty: float, px: float):
    inv_side = "Sell" if model_side=="Buy" else "Buy"
    p = pos.get(sym, {"side":inv_side, "qty":0.0, "avg":0.0})
    if p["qty"]<=0:  # 신규
        p = {"side":inv_side, "qty":qty, "avg":px}
    else:            # 같은 방향 가중평균
        if p["side"] != inv_side:
            # 반대방향이면 상계(헤지 풀기). 간단화: 먼저 기존 닫고 남은 수량만 새로 연다.
            closed = min(p["qty"], qty)
            _close_inverse(sym, closed, px)  # 현재 가격으로 상계
            remain = qty - closed
            if remain > 0:
                p = pos.get(sym, {"side":inv_side, "qty":0.0, "avg":0.0})
                if p.get("qty",0)<=0:
                    p = {"side":inv_side, "qty":remain, "avg":px}
                else:
                    p["avg"] = (p["avg"]*p["qty"] + px*remain)/(p["qty"]+remain)
                    p["qty"] += remain
            pos[sym] = p
            return
        # 같은 방향 증량
        p["avg"] = (p["avg"]*p["qty"] + px*qty)/(p["qty"]+qty)
        p["qty"] += qty
    pos[sym] = p
    _log_trade({"ts":int(time.time()),"event":"ENTRY","symbol":sym,"side":inv_side,
                "qty":f"{qty:.8f}","entry":f"{px:.6f}","exit":"","tp1":"","tp2":"","tp3":"","sl":"","reason":"mirror","mode":"inverse"})
    if VERBOSE: print(f"[OPEN_INV] {sym} {inv_side} qty={qty} px={px}")

def _close_inverse(sym: str, qty: float, px: float):
    global _paper_eq
    p = pos.get(sym);
    if not p or qty<=0 or p["qty"]<=0: return
    qty = min(qty, p["qty"])
    # inverse 포지션 PnL
    if p["side"]=="Sell":
        pnl = (p["avg"] - px) * qty
    else:
        pnl = (px - p["avg"]) * qty
    fee = abs(p["avg"]*qty*TAKER_FEE) + abs(px*qty*TAKER_FEE)
    _paper_eq += (pnl - fee)
    p["qty"] -= qty
    if p["qty"]<=0: pos.pop(sym, None)
    else: pos[sym]=p
    _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":("Buy" if p["side"]=="Sell" else "Sell"),
                "qty":f"{qty:.8f}","entry":f"{p['avg']:.6f}","exit":f"{px:.6f}",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":"mirror_close","mode":"inverse"})
    if VERBOSE: print(f"[CLOSE_INV] {sym} qty={qty} px={px} pnl={pnl:.6f} fee={fee:.6f} eq={_paper_eq:.2f}")

def equity_poller():
    global _paper_eq
    while True:
        try:
            if MIRROR_TARGET=="live":
                try:
                    wb = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
                    rows = (wb.get("result") or {}).get("list") or []
                    eq = 0.0
                    if rows:
                        for c in rows[0].get("coin", []):
                            if c.get("coin")=="USDT":
                                eq += float(c.get("walletBalance") or 0.0)
                                eq += float(c.get("unrealisedPnl") or 0.0)
                    _log_equity(eq)
                except Exception:
                    if VERBOSE: print("[EQUITY][ERR]"); traceback.print_exc(limit=1)
            else:
                _log_equity(_paper_eq)
        finally:
            time.sleep(EQUITY_PRINT_SEC)

PROCESSED = set()

def tail_and_mirror():
    last_size = 0
    while True:
        try:
            if not os.path.exists(MIRROR_SOURCE_CSV):
                time.sleep(0.5); continue
            size = os.path.getsize(MIRROR_SOURCE_CSV)
            if size == last_size:
                time.sleep(0.5); continue
            last_size = size

            with open(MIRROR_SOURCE_CSV,"r",newline="",encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    ev = (r.get("event") or "").upper()
                    sym = r.get("symbol","").strip()
                    side0 = r.get("side","").strip()
                    qty = float(r.get("qty") or 0.0)
                    en = float(r.get("entry") or 0.0)
                    ex = float(r.get("exit") or 0.0)
                    ts = int(float(r.get("ts") or 0))
                    key = (ts,sym,ev,round(qty,8),round(en,6),round(ex,6))
                    if key in PROCESSED: continue
                    PROCESSED.add(key)
                    if not sym or qty<=0: continue

                    if ev=="ENTRY":
                        if MIRROR_TARGET=="live":
                            inv_side = "Sell" if side0=="Buy" else "Buy"
                            fill = place_market_live(sym, inv_side, qty)
                            if fill is None:
                                if VERBOSE: print(f"[LIVE][FAIL] {sym} {inv_side} {qty}")
                                continue
                            _open_inverse(sym, side0, qty, fill)
                        else:
                            _open_inverse(sym, side0, qty, en)

                    elif ev in ("TP1_FILL","TP2_FILL","TP3_FILL","EXIT"):
                        # model이 일부/전체 청산 → inverse는 해당 수량만큼 반대로 닫음
                        close_qty = qty
                        if MIRROR_TARGET=="live":
                            # 실거래: inverse 포지션 반대주문으로 닫기
                            inv_side_to_close = "Buy" if (pos.get(sym,{}).get("side")=="Sell") else "Sell"
                            fill = place_market_live(sym, inv_side_to_close, close_qty)
                            if fill is None:
                                if VERBOSE: print(f"[LIVE][CLOSE_FAIL] {sym} {close_qty}")
                                continue
                            _close_inverse(sym, close_qty, fill)
                        else:
                            px = ex if ex>0 else en  # TP 로그에 exit가 없으면 entry로 대체
                            _close_inverse(sym, close_qty, px)

        except Exception:
            if VERBOSE: print("[MIRROR][ERR]"); traceback.print_exc(limit=1)
            time.sleep(0.7)

def main():
    print(f"[START] mirror_inverse | target={MIRROR_TARGET} | source={MIRROR_SOURCE_CSV} | testnet={TESTNET}")
    _ensure_csvs()
    t = threading.Thread(target=equity_poller, daemon=True); t.start()
    tail_and_mirror()

if __name__=="__main__":
    main()
