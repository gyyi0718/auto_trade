# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — PAPER-Equivalent (TP/SL 확장판)
- 진입: DeepAR 신호 (ROI 필터 없음)
- 청산:
  * TP1/TP2 분할 익절, TP3 전체 청산
  * SL (stop loss)
  * TP2 체결 후 BE 이동
  * TP1 또는 TP2 체결 이후 트레일링 SL
  * 절대익절 USD (ABS_TP_USD)
  * 변동성 기반 타임아웃(HORIZON_TIMEOUT)
"""

import os, time, math, threading, csv
from typing import Dict, Tuple,Optional
from concurrent.futures import ThreadPoolExecutor
_ORDER_SEM = threading.Semaphore(1)

# ========= ENV =========
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE","model")
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,AVMTUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")

LEVERAGE = float(os.getenv("LEVERAGE","10"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))

TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))

TP1_BPS = float(os.getenv("TP1_BPS","50.0"))
TP2_BPS = float(os.getenv("TP2_BPS","100.0"))
TP3_BPS = float(os.getenv("TP3_BPS","200.0"))
TP1_RATIO = float(os.getenv("TP1_RATIO","0.40"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.35"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))  # 1=TP1후, 2=TP2후

ABS_TP_USD = float(os.getenv("ABS_TP_USD","10.0"))

TIMEOUT_MODE = os.getenv("TIMEOUT_MODE","atr")
VOL_WIN = int(os.getenv("VOL_WIN","60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K","1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN","3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX","20"))
_v=os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS=float(_v) if (_v and _v.strip()!="") else None
PRED_HORIZ_MIN=int(os.getenv("PRED_HORIZ_MIN","1"))

MAX_OPEN=int(os.getenv("MAX_OPEN","10"))
COOLDOWN_SEC=int(os.getenv("COOLDOWN_SEC","5"))
SCAN_PARALLEL=str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS=int(os.getenv("SCAN_WORKERS","8"))
VERBOSE=str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

TESTNET=str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = os.getenv("BYBIT_API_KEY","Dlp4eJD6YFmO99T8vC")
API_SECRET = os.getenv("BYBIT_API_SECRET","YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U")

TRADES_CSV=os.getenv("TRADES_CSV","live_trades_tp_sl.csv")
EQUITY_CSV=os.getenv("EQUITY_CSV","live_equity_tp_sl.csv")

# ========= IO =========
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]
import time, random, math, threading

# === Token Bucket ===
class _Bucket:
    def __init__(self, rate_per_s: float, burst: int):
        self.rate = max(0.1, rate_per_s)   # 초당 허용 요청
        self.burst = max(1, burst)         # 버스트 허용치
        self.tokens = float(burst)
        self.last = time.time()
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            now = time.time()
            self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1.0:
                need = (1.0 - self.tokens) / self.rate
                time.sleep(max(need, 0))
                self.tokens = max(0.0, self.tokens - 1.0)  # consume
            else:
                self.tokens -= 1.0

# 전역 버킷 (엔드포인트 2개로 나눠 제어)
_BUCKET_ORDER = _Bucket(rate_per_s=float(os.getenv("RL_ORDERS_PER_S","2.0")), burst=int(os.getenv("RL_ORDERS_BURST","4")))
_BUCKET_DATA  = _Bucket(rate_per_s=float(os.getenv("RL_DATA_PER_S","5.0")),   burst=int(os.getenv("RL_DATA_BURST","8")))

# === 공통 API 호출 래퍼 (지수 백오프 + 지터) ===
def _api_call(kind: str, fn, *args, **kwargs):
    # kind: "order" | "data"
    bucket = _BUCKET_ORDER if kind=="order" else _BUCKET_DATA  # <-- 공백 제거: _BUCKET_ORDER
    max_retries = int(os.getenv("API_MAX_RETRIES","5"))
    base = float(os.getenv("API_BACKOFF_BASE","0.5"))  # 초
    for i in range(max_retries):
        try:
            bucket.take()
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            # Bybit 10006 (rate), 10002/10016 (일시), 110043(레버리지 변경중) 등만 재시도
            transient = ("10006" in msg) or ("Too many visits" in msg) or ("timeout" in msg.lower()) or ("temporarily" in msg.lower()) or ("10002" in msg) or ("10016" in msg) or ("429" in msg)
            if i == max_retries-1 or not transient:
                raise
            sleep_s = base * (2 ** i) + random.uniform(0, 0.2)
            time.sleep(sleep_s)
# === 간단 캐시 ===
class _TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self.lock = threading.Lock()
        self.data = {}
    def get(self, key):
        with self.lock:
            v = self.data.get(key)
            if not v: return None
            val, exp = v
            if time.time() > exp:
                self.data.pop(key, None); return None
            return val
    def set(self, key, val):
        with self.lock:
            self.data[key] = (val, time.time()+self.ttl)

_POS_CACHE = _TTLCache(ttl=float(os.getenv("TTL_POS_S","0.8")))
_Q_CACHE   = _TTLCache(ttl=float(os.getenv("TTL_Q_S","0.5")))
_K_CACHE   = _TTLCache(ttl=float(os.getenv("TTL_K_S","2.5")))


def _build_trade_state(entry: float, qty: float, side: str) -> dict:
    return {
        "tp1": tp_from_bps(entry, TP1_BPS, side),
        "tp2": tp_from_bps(entry, TP2_BPS, side),
        "tp3": tp_from_bps(entry, TP3_BPS, side),
        "sl":  entry*(1-SL_ROI_PCT) if side=="Buy" else entry*(1+SL_ROI_PCT),
        "tp1_filled": False,
        "tp2_filled": False,
        "be_moved":   False,
        "qty":  float(qty),
        "entry":float(entry),
        "side": side,
    }

def init_trade_state_from_positions(symbol: str) -> bool:
    pos = get_positions().get(symbol)
    if not pos: return False
    side = pos["side"]
    entry = float(pos.get("entry") or 0.0)
    qty   = float(pos.get("qty")   or 0.0)
    if entry<=0 or qty<=0: return False
    TRADE_STATE[symbol] = _build_trade_state(entry, qty, side)
    DEADLINE[symbol]    = time.time() + _calc_timeout_minutes(symbol)*60
    return True


def _decimals_from_step(step: float) -> int:
    s = f"{step:.16f}".rstrip("0")
    return 0 if "." not in s else len(s.split(".")[1])

def _fmt_to_step(x: float, step: float) -> str:
    # 스텝에 맞게 내림 + 스텝 자릿수로 포맷
    if step <= 0:
        return f"{x:.8f}".rstrip("0").rstrip(".")
    q = math.floor(x / step) * step
    d = _decimals_from_step(step)
    s = f"{q:.{max(d,0)}f}"
    return s


def _round_down(x: float, step: float) -> float:
    if step <= 0: return x
    return math.floor(x/step)*step

def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0: return x
    return math.floor(x/tick)*tick

def _fmt_qty(q: float, step: float) -> str:
    s = str(step)
    dec = 0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{q:.{dec}f}" if dec>0 else str(int(round(q)))).rstrip("0").rstrip(".")

def _fmt_px(p: float, tick: float) -> str:
    s = str(tick)
    dec = 0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{p:.{dec}f}" if dec>0 else str(int(round(p)))).rstrip("0").rstrip(".")

def place_reduce_limit(symbol: str, side: str, qty: float, price: float) -> bool:
    try:
        rule = get_instrument_rule(symbol)
        qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
        tick = float(rule["tickSize"])

        q = math.floor(qty / qty_step) * qty_step
        if q < min_qty:
            print(f"[SKIP][reduce] {symbol}: q<{min_qty} (q={q})"); return False

        px = math.floor(price / tick) * tick
        q_str = _fmt_to_step(q, qty_step)
        px_str = _fmt_to_step(px, tick)

        client.place_order(
            category=CATEGORY, symbol=symbol, side=side,
            orderType="Limit", qty=q_str, price=px_str,
            timeInForce="PostOnly", reduceOnly=True
        )
        return True
    except Exception as e:
        print(f"[ERR] place_reduce_limit {symbol} {side} {qty}@{price} -> {e}")
        return False

def set_stop_with_retry(symbol: str, sl_price: float = None, tp_price: float = None, position_idx: int = None) -> bool:
    rule = get_instrument_rule(symbol)
    params = dict(category=CATEGORY, symbol=symbol, tpslMode="Full", slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice")
    if sl_price is not None:
        sl = _round_to_tick(float(sl_price), rule["tickSize"])
        params["stopLoss"] = _fmt_px(sl, rule["tickSize"])
        params["slOrderType"] = "Market"
    if tp_price is not None:
        tp = _round_to_tick(float(tp_price), rule["tickSize"])
        params["takeProfit"] = _fmt_px(tp, rule["tickSize"])
        params["tpOrderType"] = "Market"
    if position_idx in (1, 2):
        params["positionIdx"] = position_idx
    try:
        return client.set_trading_stop(**params)["retCode"] == 0
    except Exception as e:
        print(f"[ERR] set_trading_stop {symbol} -> {e}")
        return False

def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(EQUITY_FIELDS)

def _log_trade(row:dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])

def _log_equity(eq:float):
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([int(time.time()), f"{eq:.6f}"])
    if VERBOSE: print(f"[EQUITY] {eq:.2f}")

# ========= Bybit HTTP =========
from pybit.unified_trading import HTTP
client=HTTP(api_key=API_KEY,api_secret=API_SECRET,testnet=TESTNET,timeout=10,recv_window=5000)
mkt=HTTP(testnet=TESTNET,timeout=10,recv_window=5000)

def get_quote(symbol:str)->Tuple[float,float,float]:
    key=f"q:{symbol}"
    v=_Q_CACHE.get(key)
    if v: return v
    r=_api_call("data", mkt.get_tickers, category=CATEGORY, symbol=symbol)
    row=((r.get("result") or {}).get("list") or [{}])[0]
    bid=float(row.get("bid1Price") or 0.0); ask=float(row.get("ask1Price") or 0.0); last=float(row.get("lastPrice") or 0.0)
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    out=(bid,ask,mid or 0.0)
    _Q_CACHE.set(key,out)
    return out

def _recent_1m_df(symbol: str, minutes: int = 200):
    key=f"k:{symbol}:{minutes}"
    v=_K_CACHE.get(key)
    if v is not None: return v
    r=_api_call("data", mkt.get_kline, category="linear", symbol=symbol, interval="1", limit=min(minutes,1000))
    rows=(r.get("result") or {}).get("list") or []
    if not rows:
        _K_CACHE.set(key, None); return None
    rows=rows[::-1]
    import pandas as pd, numpy as np
    df=pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "series_id": symbol,
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    df["log_return"]=np.log(df["close"]).diff().fillna(0.0)
    _K_CACHE.set(key, df)
    return df

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
    except: return 0.0

def get_positions()->Dict[str,dict]:
    v=_POS_CACHE.get("pos")
    if v: return v
    res=_api_call("data", client.get_positions, category=CATEGORY, settleCoin="USDT")
    lst=(res.get("result") or {}).get("list") or []
    out={}
    for p in lst:
        sym=p.get("symbol"); size=float(p.get("size") or 0.0)
        if not sym or size<=0: continue
        out[sym]={"side":p.get("side"),"qty":size,"entry":float(p.get("avgPrice") or 0.0)}
    _POS_CACHE.set("pos", out)
    return out

# instruments는 자주 안 바뀌니 넉넉히 캐시
_RULE_CACHE = _TTLCache(ttl=float(os.getenv("TTL_RULE_S","600")))
def get_instrument_rule(symbol:str)->Dict[str,float]:
    v=_RULE_CACHE.get(symbol)
    if v: return v
    info=_api_call("data", client.get_instruments_info, category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst:
        rules={"tickSize":0.0001,"qtyStep":0.001,"minOrderQty":0.001}
    else:
        it=lst[0]
        tick=float((it.get("priceFilter") or {}).get("tickSize") or 0.0001)
        lot =float((it.get("lotSizeFilter") or {}).get("qtyStep") or 0.001)
        minq=float((it.get("lotSizeFilter") or {}).get("minOrderQty") or lot)
        rules={"tickSize":tick,"qtyStep":lot,"minOrderQty":minq}
    _RULE_CACHE.set(symbol, rules)
    return rules

def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        return _api_call("order", client.place_order, **params)
    except Exception as e:
        msg=str(e)
        if "position idx not match position mode" in msg and side in ("Buy","Sell"):
            p2=dict(params); p2["positionIdx"]=1 if side=="Buy" else 2
            return _api_call("order", client.place_order, **p2)
        raise
def compute_qty(symbol: str, mid: float, equity: float) -> float:
    """
    가용마진(available)과 레버리지로 최대 가능 수량 한도를 잡고,
    규칙에 맞춰 라운딩. 기본 목표는 available * ENTRY_EQUITY_PCT만큼의 증거금 사용.
    """
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"])
    min_qty  = float(rule["minOrderQty"])

    if mid <= 0: return 0.0

    avail = _get_available_balance()                 # 가용 USDT
    if avail <= 0: return 0.0

    # 목표 초기증거금 = avail * ENTRY_EQUITY_PCT (너무 타이트하지 않게 0.9 안전계수)
    target_im = avail * ENTRY_EQUITY_PCT * 0.9

    # 필요증거금(IM) ≈ (mid * qty) / LEVERAGE  => qty = target_im * LEVERAGE / mid
    raw_qty = (target_im * LEVERAGE) / mid

    # 거래소 스텝에 맞게 내림
    qty = math.floor(raw_qty / qty_step) * qty_step

    # 최소수량 보정
    if qty < min_qty:
        qty = math.ceil(min_qty / qty_step) * qty_step

    return float(qty)
def place_market(symbol: str, side: str, qty: float) -> bool:
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"])
    min_qty  = float(rule["minOrderQty"])

    # 규칙 정규화
    q = math.floor(float(qty) / qty_step) * qty_step
    if q < min_qty:
        if VERBOSE: print(f"[SKIP] {symbol}: qty<{min_qty} (q={q})")
        return False

    # 문자열 포맷
    def _fmt(qv: float) -> str:
        return (f"{qv:.10f}".rstrip("0").rstrip(".")) if not qty_step.is_integer() else str(int(round(qv)))

    # 동시성 제한 (과호출 방지)
    if not _ORDER_SEM.acquire(timeout=2.0):
        return False
    try:
        # 최대 5회 시도: 110007 이면 0.74배씩 축소, 10006 은 버킷/백오프가 처리
        attempts = 0
        while q >= min_qty and attempts < 5:
            try:
                _api_call("order", client.place_order,
                          category=CATEGORY, symbol=symbol, side=side,
                          orderType="Market", qty=_fmt(q), timeInForce="IOC", reduceOnly=False)
                return True
            except Exception as e:
                msg = str(e)
                if "ErrCode: 110007" in msg or "not enough" in msg:
                    # 가용 마진 부족 → 수량 축소 후 재시도
                    q = math.floor((q * 0.74) / qty_step) * qty_step
                    attempts += 1
                    continue
                if "ErrCode: 10006" in msg or "Too many visits" in msg:
                    # 레이트리밋: 짧게 쉬고 1회만 재시도(버킷/백오프 이미 적용)
                    time.sleep(0.6)
                    attempts += 1
                    continue
                if VERBOSE: print(f"[ERR] place_market {symbol} {side} q={q} -> {e}")
                return False

        if VERBOSE: print(f"[ERR] place_market {symbol} {side}: all retries failed (last q={q})")
        return False
    finally:
        _ORDER_SEM.release()

def close_market(symbol: str, side: str, qty: float):
    try:
        rule = get_instrument_rule(symbol)
        step = float(rule["qtyStep"]); minq = float(rule["minOrderQty"])
        pos = get_positions().get(symbol)
        if not pos:
            return
        pos_qty = float(pos.get("qty") or 0.0)
        if pos_qty <= 0:
            return

        # 요청 수량을 포지션 잔량 한도로 제한 후 스텝 내림
        target = min(float(qty), pos_qty)
        q = math.floor(target / step) * step

        # minOrderQty 미만이면 실행 금지 (거래소가 거부함)
        if q < minq:
            if VERBOSE:
                print(f"[SKIP][close] {symbol}: q<{minq} (q={q}), pos_qty={pos_qty}  -> skip")
            return

        # 최종 문자열 포맷
        def _fmt(qv: float) -> str:
            s = f"{step:.16f}".rstrip("0")
            dec = 0 if "." not in s else len(s.split(".")[1])
            return (f"{qv:.{dec}f}" if dec>0 else str(int(round(qv)))).rstrip("0").rstrip(".")

        close_side = "Sell" if side == "Buy" else "Buy"
        _api_call("order", client.place_order,
                  category=CATEGORY, symbol=symbol, side=close_side,
                  orderType="Market", qty=_fmt(q), timeInForce="IOC", reduceOnly=True)
    except Exception as e:
        print(f"[ERR] close {symbol} {('Sell' if side=='Buy' else 'Buy')} {qty} -> {e}")

# ========= Timeout =========
import numpy as np, pandas as pd
def _recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes=minutes or VOL_WIN
    r=mkt.get_kline(category=CATEGORY,symbol=symbol,interval="1",limit=min(minutes,1000))
    rows=(r.get("result") or {}).get("list") or []
    if len(rows)<10: return 10.0
    closes=[float(z[4]) for z in rows][::-1]
    lr=np.diff(np.log(np.asarray(closes))); v_bps=float(np.median(np.abs(lr))*1e4)
    return max(v_bps,1.0)

def _calc_timeout_minutes(symbol:str)->int:
    if TIMEOUT_MODE=="fixed": return max(1,int(PRED_HORIZ_MIN))
    target_bps=TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps=_recent_vol_bps(symbol,VOL_WIN)
    need=int(math.ceil((target_bps/v_bps)*TIMEOUT_K))
    return int(min(max(need,TIMEOUT_MIN),TIMEOUT_MAX))

# ========= State =========
COOLDOWN_UNTIL:Dict[str,float]={}
DEADLINE:Dict[str,float]={}
TRADE_STATE:Dict[str,dict]={}  # TP/SL/BE 상태 추적
def _get_available_balance() -> float:
    """
    UNIFIED 계정 USDT 가용 잔고 (availableBalance + unrealisedPnl 유사) 추정
    """
    try:
        wb = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows = (wb.get("result") or {}).get("list") or []
        if not rows: return 0.0
        c = next((x for x in rows[0].get("coin", []) if x.get("coin") == "USDT"), None)
        if not c: return 0.0
        # availableToWithdraw가 없을 수 있어 보수적으로 walletBalance + unrealisedPnl 사용
        bal = float(c.get("walletBalance") or 0.0) + float(c.get("unrealisedPnl") or 0.0)
        return max(bal, 0.0)
    except:
        return 0.0
# ========= Entry =========
def try_enter(symbol:str):
    now = time.time()
    if COOLDOWN_UNTIL.get(symbol,0.0) > now:
        return

    pos = get_positions()  # 캐시 있음
    if symbol in pos:
        return
    if len(pos) >= MAX_OPEN:
        return

    # 주문 버킷이 바닥이면 스킵 (추가 보호)
    if _BUCKET_ORDER.tokens < 0.9:
        return

    # === 신호 (임시 랜덤) ===
    side = "Buy" if int(now*1000) % 2 == 0 else "Sell"

    bid, ask, mid = get_quote(symbol)
    if mid <= 0:
        return

    eq = get_wallet_equity()
    qty = compute_qty(symbol, mid, eq)
    if qty <= 0:
        return

    ok = place_market(symbol, side, qty)
    if not ok:
        return

    # 체결 포지션 조회
    pos = get_positions().get(symbol)
    if not pos:
        return
    entry = float(pos.get("entry") or 0.0) or mid
    pidx = pos.get("positionIdx") if isinstance(pos, dict) else None

    # ⬇️ 여기 추가: 거래소 TP/SL 부착
    attach_tp_sl(symbol, side, entry, float(pos.get("qty") or qty), position_idx=pidx)

    # 타임아웃 마감 시각 저장(있다면)
    DEADLINE[symbol] = time.time() + _calc_timeout_minutes(symbol) * 60

    # 1순위: 체결 포지션으로 상태 구성
    if not init_trade_state_from_positions(symbol):
        # 2순위: 현재 mid 기반 fallback
        horizon_min = _calc_timeout_minutes(symbol)
        DEADLINE[symbol] = now + horizon_min * 60
        TRADE_STATE[symbol] = _build_trade_state(mid, qty, side)

    _log_trade({
        "ts": int(now), "event": "ENTRY", "symbol": symbol, "side": side,
        "qty": f"{qty:.8f}", "entry": f"{mid:.6f}", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
        "reason": "open", "mode": ENTRY_MODE
    })
    if VERBOSE:
        print(f"[OPEN] {symbol} {side} {qty}@{mid:.4f}")

def tp_from_bps(entry:float,bps:float,side:str)->float:
    return entry*(1+bps/10000.0) if side=="Buy" else entry*(1-bps/10000.0)

def attach_tp_sl(symbol: str, side: str, entry: float, qty: float, position_idx: Optional[int] = None):
    rule = get_instrument_rule(symbol)
    lot = float(rule["qtyStep"]); minq = float(rule["minOrderQty"])
    close_side = "Sell" if side == "Buy" else "Buy"

    def _rd(x, step):  # 스텝 내림
        return math.floor(x/step)*step

    # 목표가
    tp1 = tp_from_bps(entry, TP1_BPS, side)
    tp2 = tp_from_bps(entry, TP2_BPS, side)
    tp3 = tp_from_bps(entry, TP3_BPS, side)
    sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)

    # 분할수량(스텝/최소수량 보정)
    q1 = _rd(qty*TP1_RATIO, lot)
    q2 = _rd(qty*TP2_RATIO, lot)

    # TP1/TP2: reduceOnly PostOnly 지정가
    if q1 >= minq:
        place_reduce_limit(symbol, close_side, q1, tp1)
    if q2 >= minq:
        place_reduce_limit(symbol, close_side, q2, tp2)

    # TP3/SL: trading_stop(Full)
    set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=position_idx)

    # 상태 초기화(옵션)
    TRADE_STATE[symbol] = {
        "side": side, "entry": float(entry), "qty": float(qty),
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
        "tp1_filled": False, "tp2_filled": False, "be_moved": False
    }


# ========= Monitor =========
def monitor_loop():
    while True:
        try:
            _log_equity(get_wallet_equity())
            pos=get_positions(); now=time.time()
            # 신규 포지션 세팅
            new_syms = set(pos.keys()) - set(TRADE_STATE.keys())
            for sym in new_syms:
                p = pos[sym]
                side = p["side"]
                entry = float(p.get("entry") or 0.0)
                qty = float(p.get("qty") or 0.0)
                if entry <= 0 or qty <= 0:
                    continue
                attach_tp_sl(sym, side, entry, qty, position_idx=p.get("positionIdx"))
                DEADLINE[sym] = time.time() + _calc_timeout_minutes(sym) * 60
                st = TRADE_STATE.get(sym)
                if not st:
                    if not init_trade_state_from_positions(sym):
                        continue
                    st = TRADE_STATE.get(sym)
                    TRADE_STATE[sym] = {
                        "side": p.get("side", "Buy"),
                        "entry": float(p.get("entry") or 0.0),
                        "qty": float(p.get("qty") or 0.0),
                        "tp1_filled": False, "tp2_filled": False, "be_moved": False
                    }

                # 누락 키 보정
                st.setdefault("side", p.get("side", "Buy"))
                st.setdefault("entry", float(p.get("entry") or 0.0))
                st.setdefault("qty", float(p.get("qty") or 0.0))

                _, _, mark_now = get_quote(sym)
                ref = st["entry"] if st["entry"] > 0 else (mark_now or 0.0)

                def _tp(entry, bps, side):
                    return entry * (1 + bps / 10000.0) if side == "Buy" else entry * (1 - bps / 10000.0)

                if "tp1" not in st: st["tp1"] = _tp(ref, TP1_BPS, st["side"])
                if "tp2" not in st: st["tp2"] = _tp(ref, TP2_BPS, st["side"])
                if "tp3" not in st: st["tp3"] = _tp(ref, TP3_BPS, st["side"])
                if "sl" not in st:
                    st["sl"] = ref * (1 - SL_ROI_PCT) if st["side"] == "Buy" else ref * (1 + SL_ROI_PCT)

                st.setdefault("tp1_filled", False)
                st.setdefault("tp2_filled", False)
                st.setdefault("be_moved", False)

                # 누락 키 보정
                if "qty" not in st: st["qty"] = float(p.get("qty") or 0.0)
                if "side" not in st: st["side"] = p.get("side", "Buy")
                if "entry" not in st: st["entry"] = float(p.get("entry") or 0.0)
                side=st["side"]; entry=st["entry"]; qty=st["qty"]; tp1, tp2, tp3, sl=st["tp1"],st["tp2"],st["tp3"],st["sl"]
                _,_,mark=get_quote(sym)
                if mark<=0: continue

                # --- 규칙 불러오기 (루프 상단, sym별로 1회만) ---
                rule = get_instrument_rule(sym)
                step = float(rule["qtyStep"]);
                minq = float(rule["minOrderQty"])

                # TP1
                if (not st["tp1_filled"]) and ((side == "Buy" and mark >= tp1) or (side == "Sell" and mark <= tp1)):
                    part = math.floor((qty * TP1_RATIO) / step) * step
                    if part >= minq:
                        close_market(sym, side, part)
                        st["tp1_filled"] = True
                        st["qty"] = qty = max(0.0, qty - part)
                        _log_trade({"ts": int(time.time()), "event": "TP1_FILL", "symbol": sym, "side": side,
                                    "qty": f"{part:.8f}", "entry": f"{entry:.6f}", "exit": f"{tp1:.6f}",
                                    "tp1": "", "tp2": "", "tp3": "", "sl": "", "reason": "", "mode": "paper"})

                # TP2
                if (st["qty"] > 0) and (not st["tp2_filled"]) and (
                        (side == "Buy" and mark >= tp2) or (side == "Sell" and mark <= tp2)):
                    # 남은 잔량 기준으로 재계산
                    part = math.floor((st["qty"] * TP2_RATIO) / step) * step
                    # 너무 작으면 남은 전량(스텝 내림)으로 시도
                    if part < minq:
                        part = math.floor(st["qty"] / step) * step
                    if part >= minq and part <= st["qty"]:
                        close_market(sym, side, part)
                        st["tp2_filled"] = True
                        st["qty"] = qty = max(0.0, st["qty"] - part)
                        _log_trade({"ts": int(time.time()), "event": "TP2_FILL", "symbol": sym, "side": side,
                                    "qty": f"{part:.8f}", "entry": f"{entry:.6f}", "exit": f"{tp2:.6f}",
                                    "tp1": "", "tp2": "", "tp3": "", "sl": "", "reason": "", "mode": "paper"})

                # BE
                if st["tp2_filled"] and not st["be_moved"]:
                    st["sl"]=entry*(1+BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1-BE_EPS_BPS/10000.0)
                    st["be_moved"]=True
                    _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":f"{st['sl']:.6f}","reason":"BE","mode":"paper"})

                # Trailing
                if (st["tp1_filled"] if TRAIL_AFTER_TIER==1 else st["tp2_filled"]):
                    if side=="Buy": st["sl"]=max(st["sl"], mark*(1-TRAIL_BPS/10000.0))
                    else: st["sl"]=min(st["sl"], mark*(1+TRAIL_BPS/10000.0))

                # Exit: SL, TP3, ABS_TP, Timeout
                pnl=(mark-entry)*qty if side=="Buy" else (entry-mark)*qty
                hit_abs=pnl>=ABS_TP_USD
                hit_tp3=(side=="Buy" and mark>=tp3) or (side=="Sell" and mark<=tp3)
                hit_sl=(side=="Buy" and mark<=st["sl"]) or (side=="Sell" and mark>=st["sl"])
                hit_to=(now>=DEADLINE.get(sym,now+1e9))

                if hit_abs or hit_tp3 or hit_sl or hit_to:
                    close_market(sym,side,st["qty"])
                    _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":side,"qty":f"{st['qty']:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}","tp1":"","tp2":"","tp3":"","sl":"","reason":"ABS_TP" if hit_abs else ("TP3" if hit_tp3 else ("SL" if hit_sl else "HORIZON_TIMEOUT")),"mode":"paper"})
                    COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC; TRADE_STATE.pop(sym,None); DEADLINE.pop(sym,None)

            time.sleep(1)
        except Exception as e: print("[MONITOR ERR]",e); time.sleep(2)

# ========= Scanner =========
def scanner_loop():
    idx = 0
    while True:
        try:
            pos = get_positions()
            free_slots = max(0, MAX_OPEN - len(pos))
            if free_slots <= 0:
                time.sleep(1)
                continue

            # 이번 루프에서 시도할 소수만 선택
            batch = SYMBOLS[idx:idx+free_slots]
            if not batch:
                idx = 0
                batch = SYMBOLS[:free_slots]
            idx = (idx + free_slots) % len(SYMBOLS)

            if SCAN_PARALLEL and free_slots > 1:
                with ThreadPoolExecutor(max_workers=min(SCAN_WORKERS, free_slots)) as ex:
                    ex.map(try_enter, batch)
            else:
                for s in batch:
                    try_enter(s)

            time.sleep(1.5)
        except Exception as e:
            print("[SCANNER ERR]", e)
            time.sleep(2)

# ========= Main =========
def main():
    print(f"[START] PAPER-Equivalent LIVE EXTENDED | MODE={ENTRY_MODE} TESTNET={TESTNET}")
    t1=threading.Thread(target=monitor_loop,daemon=True); t2=threading.Thread(target=scanner_loop,daemon=True)
    t1.start(); t2.start()
    while True: time.sleep(60)

if __name__=="__main__":
    main()
