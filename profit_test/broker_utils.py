# broker_utils.py
import os, time, math, json
from typing import Dict, Tuple, Optional, List
from pybit.unified_trading import HTTP

CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")
TESTNET  = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)

def get_instrument_rule(symbol:str)->Dict[str,float]:
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst  = (info.get("result") or {}).get("list") or []
    if not lst:
        return {"tickSize":0.0001,"qtyStep":0.001,"minOrderQty":0.001}
    it   = lst[0]
    tick = float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
    lot  = float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
    minq = float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
    return {"tickSize":tick,"qtyStep":lot,"minOrderQty":minq}

def _fmt_qty(q: float, step: float) -> str:
    s = f"{step:.16f}".rstrip("0")
    dec = 0 if "." not in s else len(s.split(".")[1])
    return (f"{q:.{dec}f}" if dec>0 else str(int(round(q)))).rstrip("0").rstrip(".")

def get_positions()->Dict[str,dict]:
    res = client.get_positions(category=CATEGORY, settleCoin="USDT")
    lst = (res.get("result") or {}).get("list") or []
    out={}
    for p in lst:
        sym  = p.get("symbol"); size=float(p.get("size") or 0.0)
        if not sym or size<=0: continue
        out[sym] = {
            "side":p.get("side"),
            "qty": size,
            "positionIdx": int(p.get("positionIdx") or 0),
        }
    return out

def cancel_all_open_orders(symbol: Optional[str]=None):
    try:
        client.cancel_all_orders(category=CATEGORY, symbol=symbol)  # 전체 취소
    except Exception:
        pass

def close_all_positions(reduce_batch_sleep: float = 0.25):
    pos = get_positions()
    for sym, p in pos.items():
        side = p["side"]
        qty  = float(p["qty"] or 0.0)
        if qty <= 0: continue
        rule = get_instrument_rule(sym)
        lot  = float(rule["qtyStep"]); minq=float(rule["minOrderQty"])
        q    = math.floor(qty/lot)*lot
        if q < minq: continue
        close_side = "Sell" if side=="Buy" else "Buy"
        try:
            client.place_order(
                category=CATEGORY, symbol=sym, side=close_side,
                orderType="Market", qty=_fmt_qty(q, lot),
                timeInForce="IOC", reduceOnly=True
            )
        except Exception:
            # 개별 종목 실패는 무시하고 진행
            pass
        time.sleep(reduce_batch_sleep)

def get_wallet_equity()->float:
    try:
        wb=client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows=(wb.get("result") or {}).get("list") or []
        total=0.0
        for c in rows[0].get("coin",[]):
            if c.get("coin")=="USDT":
                total+=float(c.get("walletBalance") or 0.0)
                total+=float(c.get("unrealisedPnl") or 0.0)
        return float(total)
    except Exception:
        return 0.0
