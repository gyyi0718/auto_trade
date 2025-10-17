# bybit_gpt_trade.py
# -*- coding: utf-8 -*-
"""
Bybit v5 1분 규칙형 트레이더 (Paper 기본)
- 모든 폴링마다 상태 로그 출력: Long / Short / Flat
- 신호가 임계값 이상일 때만 진입
- LIVE=1이면 실주문. 기본은 PAPER 로그만 출력
"""
import os, time, math, statistics, signal
from typing import Dict, Tuple, Optional, List
from decimal import Decimal, ROUND_DOWN
import requests
from pybit.unified_trading import HTTP

# ===== env =====
def env_str(k, d=""): v=os.getenv(k); return d if v is None or str(v).strip()=="" else str(v).strip()
def env_int(k, d=0):
    v=os.getenv(k)
    try: return int(str(v).strip()) if v and str(v).strip()!="" else d
    except: return d
def env_float(k, d=0.0):
    v=os.getenv(k)
    try: return float(str(v).strip()) if v and str(v).strip()!="" else d
    except: return d

API_KEY     = env_str("BYBIT_API_KEY")
API_SECRET  = env_str("BYBIT_API_SECRET")
TESTNET     = env_int("TESTNET", 0)                  # 0=live, 1=testnet
LIVE        = env_int("LIVE", 0)                     # 0=paper
CATEGORY    = env_str("CATEGORY", "linear")
SYMBOLS     = env_str("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
INTERVAL    = env_str("INTERVAL", "1")
RANGE_BARS  = env_int("RANGE_BARS", 30)              # 레인지 산출 구간
NEAR_BPS    = env_float("NEAR_BPS", 20)              # 상하단 근접 기준(bps)
IMB_THR     = env_float("IMB_THR", 1.4)              # ask/bid 임계
MOM_N       = env_int("MOM_N", 3)                    # 최근 모멘텀 길이
COOLDOWN_S  = env_int("COOLDOWN_SEC", 30)
MAX_OPEN    = env_int("MAX_OPEN", 2)
LEVERAGE    = env_float("LEVERAGE", 10.0)
EQUITY_PCT  = env_float("ENTRY_EQUITY_PCT", 0.02)
TP_BPS      = env_float("TP_BPS", 120)
SL_BPS      = env_float("SL_BPS", 70)
SPREAD_MAX  = env_float("SPREAD_BPS_MAX", 25)
POLL_SEC    = env_int("POLL_SEC", 3)
LOG         = env_int("LOG", 1)

# 상태 분류 가중치/임계
FLAT_THR    = env_float("FLAT_THR", 0.55)            # 미만이면 Flat
W_NEAR      = env_float("W_NEAR", 0.45)
W_IMB       = env_float("W_IMB",  0.35)
W_MOM       = env_float("W_MOM",  0.20)

BASE_URL = "https://api-testnet.bybit.com" if TESTNET==1 else "https://api.bybit.com"
sess = HTTP(api_key=API_KEY if LIVE else None,
            api_secret=API_SECRET if LIVE else None,
            testnet=(TESTNET==1))

def log(*a):
    if LOG: print(*a, flush=True)

# ===== REST utils =====
def safe_get(path, params, timeout=7):
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=timeout).json()
        if r.get("retCode") != 0: return None
        return r["result"]
    except Exception:
        return None

def get_klines(symbol:str, limit:int=None)->Optional[List[Dict]]:
    limit = limit or (RANGE_BARS + 5)
    res = safe_get("/v5/market/kline",
                   {"category": CATEGORY, "symbol": symbol, "interval": INTERVAL, "limit": limit})
    if not res or not res.get("list"): return None
    rows = sorted(res["list"], key=lambda x:int(x[0]))
    out=[]
    for it in rows:
        try:
            out.append({"ts":int(it[0]),"o":float(it[1]),"h":float(it[2]),
                        "l":float(it[3]),"c":float(it[4]),"v":float(it[5])})
        except: return None
    return out

def get_orderbook(symbol:str, depth:int=25)->Optional[Tuple[float,float,float]]:
    res = safe_get("/v5/market/orderbook", {"category": CATEGORY, "symbol": symbol, "limit": depth})
    if not res or not res.get("b") or not res.get("a"): return None
    bids=[(float(p),float(q)) for p,q in res["b"] if float(q)>0]
    asks=[(float(p),float(q)) for p,q in res["a"] if float(q)>0]
    if not bids or not asks: return None
    best_bid, best_ask = bids[0][0], asks[0][0]
    wb=sum(q for _,q in bids[:10]); wa=sum(q for _,q in asks[:10])
    if wb<=0 or wa<=0: return None
    return best_bid, best_ask, wa/max(wb,1e-9)  # ask/bid

# ===== math =====
def bps(a,b):
    mid=(a+b)/2.0
    return abs(b-a)/max(mid,1e-9)*1e4

def calc_range(k:List[Dict])->Tuple[float,float,float]:
    seg=k[-RANGE_BARS:]
    hi=max(x["h"] for x in seg); lo=min(x["l"] for x in seg)
    pv=(seg[-1]["c"]+hi+lo)/3.0
    return lo,hi,pv

def recent_momentum(k:List[Dict], n:int)->float:
    closes=[x["c"] for x in k[-(n+1):]]
    if len(closes)<2: return 0.0
    rets=[(closes[i+1]-closes[i])/max(closes[i],1e-12) for i in range(len(closes)-1)]
    try: return statistics.fmean(rets)
    except: return 0.0

# ===== classifier =====
def classify_state(symbol: str):
    """롱/숏/횡보 상태를 매번 로그. score>=FLAT_THR면 진입 후보."""
    info = {"reason": "none"}
    k = get_klines(symbol)
    ob = get_orderbook(symbol)
    if not k or not ob:
        # 데이터 없을 때도 매번 로그
        print(f"[STATE][{symbol}] Flat score=0.000 near=- spr=NA imb=NA mom=NA -> no_data", flush=True)
        return "Flat", 0.0, {"reason":"no_data"}

    lo, hi, pv = calc_range(k)
    bid, ask, imb = ob
    mid = (bid + ask) / 2.0
    spr = bps(bid, ask)
    mom = recent_momentum(k, MOM_N)

    def near_score(dist_bps):
        return max(0.0, min(1.0, 1.0 - dist_bps / max(NEAR_BPS, 1e-9)))

    near_top = near_score(bps(mid, hi))
    near_bot = near_score(bps(mid, lo))

    imb_long  = max(0.0, min(1.0, (1.0/IMB_THR - imb) / (1.0/IMB_THR))) if imb <= 1.0/IMB_THR else 0.0
    imb_short = max(0.0, min(1.0, (imb - IMB_THR) / IMB_THR))            if imb >= IMB_THR      else 0.0
    mom_long  = max(0.0, min(1.0,  5000.0 * mom))
    mom_short = max(0.0, min(1.0, -5000.0 * mom))

    score_long  = W_NEAR*near_bot  + W_IMB*imb_long  + W_MOM*mom_long
    score_short = W_NEAR*near_top  + W_IMB*imb_short + W_MOM*mom_short

    if spr > SPREAD_MAX:
        label, score, reason = "Flat", 0.0, "spread"
    else:
        mscore = max(score_long, score_short)
        if mscore < FLAT_THR:
            label, score, reason = "Flat", mscore, "flat_thr"
        else:
            if score_long >= score_short:
                label, score, reason = "Long", score_long, "rule_long"
            else:
                label, score, reason = "Short", score_short, "rule_short"

    near = 'TOP' if bps(mid, hi) <= NEAR_BPS else ('BOT' if bps(mid, lo) <= NEAR_BPS else '-')
    print(f"[STATE][{symbol}] {label} score={max(score_long,score_short):.3f} "
          f"near={near} spr={spr:.1f}bps imb={imb:.2f} mom={mom:.4f} "
          f"NT={near_top:.2f} NB={near_bot:.2f} IL={imb_long:.2f} IS={imb_short:.2f} "
          f"SL={score_long:.2f} SS={score_short:.2f} -> {reason}", flush=True)

    info = dict(mid=mid, lo=lo, hi=hi, pv=pv, spr_bps=spr, ask_over_bid=imb, mom=mom,
                score_long=score_long, score_short=score_short, reason=reason)
    return label, max(score_long, score_short), info

# ===== trading utils =====
def get_wallet_balance()->float:
    try:
        r=sess.get_wallet_balance(accountType="UNIFIED")
        for a in r["result"]["list"]:
            for c in a["coin"]:
                if c["coin"]=="USDT": return float(c["walletBalance"])
    except: pass
    return 0.0

def get_instrument_info(symbol:str)->Tuple[int,float]:
    try:
        r=sess.get_instruments_info(category=CATEGORY, symbol=symbol)
        it=r["result"]["list"][0]
        step=float(it["lotSizeFilter"]["qtyStep"])
        tick=float(it["priceFilter"]["tickSize"])
        step_dec=max(0,int(round(-math.log10(step)))) if step<1 else 0
        return step_dec, tick
    except: return 0, 0.0001

def round_qty(q:float, step_dec:int)->str:
    qd=Decimal(q).quantize(Decimal(10)**-step_dec, rounding=ROUND_DOWN)
    return format(qd,"f")

def ensure_leverage(symbol:str):
    try:
        sess.set_leverage(category=CATEGORY, symbol=symbol,
                          buyLeverage=str(LEVERAGE), sellLeverage=str(LEVERAGE))
    except: pass

def last_price(symbol:str)->Optional[float]:
    try:
        r=sess.get_tickers(category=CATEGORY, symbol=symbol)
        return float(r["result"]["list"][0]["lastPrice"])
    except: return None

def price_from_bps(entry:float, bps_val:float, side:str, is_tp:bool)->float:
    if side=="Buy":
        return entry*(1+bps_val/1e4) if is_tp else entry*(1-SL_BPS/1e4)
    else:
        return entry*(1-bps_val/1e4) if is_tp else entry*(1+SL_BPS/1e4)

def place_order(symbol:str, side:str, qty:float, tp:float, sl:float)->Optional[str]:
    if LIVE!=1:
        log(f"[PAPER] {symbol} {side} qty={qty} TP={tp} SL={sl}")
        return None
    try:
        o=sess.place_order(category=CATEGORY, symbol=symbol, side=side,
                           orderType="Market", qty=str(qty),
                           reduceOnly=False, timeInForce="IOC", positionIdx=0,
                           takeProfit=str(tp), stopLoss=str(sl),
                           tpTriggerBy="LastPrice", slTriggerBy="LastPrice")
        return o["result"]["orderId"]
    except Exception as e:
        log("place_order err:", e); return None

OPEN_TS: Dict[str,float] = {}
IN_POS: Dict[str,str] = {}

def can_open(symbol:str)->bool:
    if len(IN_POS)>=MAX_OPEN and symbol not in IN_POS: return False
    if time.time()-OPEN_TS.get(symbol,0.0)<COOLDOWN_S: return False
    return True

def calc_qty_usdt(symbol:str, entry_price:float)->float:
    bal=get_wallet_balance()
    risk=max(bal*EQUITY_PCT, 5.0)     # 최소 5 USDT
    qty=risk*LEVERAGE/max(entry_price,1e-9)
    step_dec,_=get_instrument_info(symbol)
    return float(round_qty(qty, step_dec))

# ===== main =====
_terminate=False
def _sig(*_):
    global _terminate; _terminate=True
signal.signal(signal.SIGINT,_sig); signal.signal(signal.SIGTERM,_sig)
def make_signal(symbol:str):
    """완화된 규칙:
    1) 스프레드가 과하면 FLAT.
    2) 경계 근접 AND (오더북 OR 모멘텀) 이면 방향.
    3) 경계 아니어도 모멘텀 절대값이 임계 이상이면 방향.
    4) 아니면 FLAT.
    """
    MOM_THR = float(os.getenv("MOM_THR", "0.0005"))   # 0.05% 평균 수익률
    NEAR_MULT = float(os.getenv("NEAR_MULT", "1.5"))  # 경계 판정 완화 배수

    info = {"reason": "none"}
    k = get_klines(symbol)
    ob = get_orderbook(symbol)
    if not k or not ob:
        info["reason"] = "no_data"
        return None, info

    lo, hi, pv = calc_range(k)
    bid, ask, imb = ob
    mid = (bid + ask) / 2.0
    spr = bps(bid, ask)
    mom = recent_momentum(k, MOM_N)

    # 측정치 로깅용
    dist_top = bps(mid, hi)
    dist_bot = bps(mid, lo)
    near_top = dist_top <= NEAR_BPS * NEAR_MULT
    near_bot = dist_bot <= NEAR_BPS * NEAR_MULT

    info.update(dict(
        mid=mid, lo=lo, hi=hi, pv=pv,
        spr_bps=spr, ask_over_bid=imb, mom=mom,
        dist_top_bps=dist_top, dist_bot_bps=dist_bot,
        near_top=near_top, near_bot=near_bot
    ))

    # 1) 스프레드 과다면 거래 안 함
    if spr > SPREAD_MAX:
        info["reason"] = "spread"
        return None, info

    # 2) 경계 근접 + (오더북 OR 모멘텀) 신호
    #    오더북 임계 완화: 1.4 -> 1.2 기본
    IMB_SOFT = float(os.getenv("IMB_SOFT", "1.2"))
    if near_top and ((imb >= IMB_SOFT) or (mom <= -MOM_THR)):
        info["reason"] = "short_rule"
        return "Sell", info
    if near_bot and ((imb <= 1.0 / IMB_SOFT) or (mom >= MOM_THR)):
        info["reason"] = "long_rule"
        return "Buy", info

    # 3) 폴백: 경계가 아니어도 모멘텀만으로 방향
    if abs(mom) >= MOM_THR:
        side = "Buy" if mom > 0 else "Sell"
        info["reason"] = "momentum_fallback"
        return side, info

    # 4) 그 외 FLAT
    info["reason"] = "flat"
    return None, info

def main():
    print(f"[START] RULE AUTOTRADER | TESTNET={TESTNET} | LIVE={LIVE} | SYMS={SYMBOLS} | POLL_SEC={POLL_SEC}", flush=True)
    while not _terminate:
        t0 = time.time()
        try:
            for sym in SYMBOLS:
                label, score, info = classify_state(sym)  # 항상 로그 출력
                if label in ("Long","Short") and can_open(sym):
                    side  = "Buy" if label=="Long" else "Sell"
                    ensure_leverage(sym)
                    entry = last_price(sym) or info.get("mid", 0.0)
                    if entry <= 0:
                        print(f"[SKIP]{sym} no_last_price", flush=True); continue
                    tp = price_from_bps(entry, TP_BPS, side, True)
                    sl = price_from_bps(entry, SL_BPS, side, False)
                    qty = calc_qty_usdt(sym, entry)
                    if qty <= 0:
                        print(f"[SKIP]{sym} qty<=0", flush=True); continue
                    place_order(sym, side, qty, tp, sl)
                    IN_POS[sym]=side; OPEN_TS[sym]=time.time()
                    print(f"[OPEN] {sym} {side} @ {entry:.6f} qty={qty} tp={tp:.6f} sl={sl:.6f}", flush=True)
        except Exception as e:
            print("loop err:", e, flush=True)
        time.sleep(max(0.2, POLL_SEC - (time.time()-t0)))

if __name__=="__main__":
    main()
