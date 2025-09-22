# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp) — PAPER-Equivalent (TP/SL 확장판)
- 신호: DeepAR(+온라인 어댑터 보정) | ENTRY_MODE: model|inverse|random
- 진입: ROI 필터 없음(원문 유지). 수량은 가용잔고 기반.
- 청산: TP1/TP2 분할, TP3/SL trading_stop, TP2 후 BE 이동, 트레일링, ABS_TP_USD, ATR 타임아웃
"""

import os, time, math, threading, csv, random
from typing import Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# ========= ENV =========
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE","model").lower()          # model|inverse|random
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,AVMTUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")

LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.20"))

TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))  # 임계 여유

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
VOL_WIN = int(os.getenv("VOL_WIN","240"))
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

# ======== MODEL / ONLINE ADAPTER ENV ========
MODEL_MODE = os.getenv("MODEL_MODE","online").lower()     # fixed|online
MODEL_CKPT = os.getenv("MODEL_CKPT","models/multi_deepar_model.ckpt")
SEQ_LEN    = int(os.getenv("SEQ_LEN","240"))
PRED_LEN   = int(os.getenv("PRED_LEN","60"))
ONLINE_LR    = float(os.getenv("ONLINE_LR","0.05"))
ONLINE_MAXBUF= int(os.getenv("ONLINE_MAXBUF","240"))
ADAPTER_REFIT_SEC = int(os.getenv("ADAPTER_REFIT_SEC","120"))

# ========= IO =========
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]

# ========= Token Bucket / Backoff =========
import time as _t, threading as _th, random as _rd, math as _m
class _Bucket:
    def __init__(self, rate_per_s: float, burst: int):
        self.rate = max(0.1, rate_per_s)
        self.burst = max(1, burst)
        self.tokens = float(burst)
        self.last = _t.time()
        self.lock = _th.Lock()
    def take(self):
        with self.lock:
            now = _t.time()
            self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1.0:
                need = (1.0 - self.tokens) / self.rate
                _t.sleep(max(need, 0))
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

_BUCKET_ORDER = _Bucket(rate_per_s=float(os.getenv("RL_ORDERS_PER_S","2.0")), burst=int(os.getenv("RL_ORDERS_BURST","4")))
_BUCKET_DATA  = _Bucket(rate_per_s=float(os.getenv("RL_DATA_PER_S","5.0")),   burst=int(os.getenv("RL_DATA_BURST","8")))
_ORDER_SEM = threading.Semaphore(1)

def _api_call(kind: str, fn, *args, **kwargs):
    bucket = _BUCKET_ORDER if kind=="order" else _BUCKET_DATA  # <- 오타 방지용 별칭
    bucket.take()
    max_retries = int(os.getenv("API_MAX_RETRIES","5"))
    base = float(os.getenv("API_BACKOFF_BASE","0.5"))
    for i in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            transient = ("10006" in msg) or ("Too many visits" in msg) or ("timeout" in msg.lower()) or ("temporarily" in msg.lower()) or ("10002" in msg) or ("10016" in msg) or ("429" in msg)
            if i == max_retries-1 or not transient:
                raise
            _t.sleep(base * (2 ** i) + _rd.uniform(0, 0.2))

# ========= Caches =========
class _TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl; self.lock = threading.Lock(); self.data = {}
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
_RULE_CACHE= _TTLCache(ttl=float(os.getenv("TTL_RULE_S","600")))

# ========= Bybit HTTP =========
from pybit.unified_trading import HTTP
client=HTTP(api_key=API_KEY,api_secret=API_SECRET,testnet=TESTNET,timeout=10,recv_window=5000)
mkt=HTTP(testnet=TESTNET,timeout=10,recv_window=5000)

# ========= Utils =========
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

def _decimals_from_step(step: float) -> int:
    s = f"{step:.16f}".rstrip("0")
    return 0 if "." not in s else len(s.split(".")[1])
def _fmt_to_step(x: float, step: float) -> str:
    if step <= 0: return f"{x:.8f}".rstrip("0").rstrip(".")
    q = math.floor(x/step)*step
    d = _decimals_from_step(step)
    return f"{q:.{max(d,0)}f}"
def _round_down(x: float, step: float) -> float:
    return x if step<=0 else math.floor(x/step)*step
def _round_to_tick(x: float, tick: float) -> float:
    return x if tick<=0 else math.floor(x/tick)*tick
def _fmt_qty(q: float, step: float) -> str:
    s = str(step); dec = 0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{q:.{dec}f}" if dec>0 else str(int(round(q)))).rstrip("0").rstrip(".")
def _fmt_px(p: float, tick: float) -> str:
    s = str(tick); dec = 0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{p:.{dec}f}" if dec>0 else str(int(round(p)))).rstrip("0").rstrip(".")

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
        if not rows: return 0.0
        tot=0.0
        for c in rows[0].get("coin",[]):
            if c.get("coin")=="USDT":
                tot += float(c.get("walletBalance") or 0.0)
                tot += float(c.get("unrealisedPnl") or 0.0)
        return float(tot)
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
        out[sym]={"side":p.get("side"),"qty":size,"entry":float(p.get("avgPrice") or 0.0), "positionIdx":int(p.get("positionIdx") or 0)}
    _POS_CACHE.set("pos", out)
    return out

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

# ========= MODEL / ONLINE ADAPTER =========
import numpy as np, pandas as pd, collections
import torch
from torch.serialization import add_safe_globals
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR

class _CompatLoss(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self,*a,**k): return torch.tensor(0.0)
add_safe_globals([GroupNormalizer, _CompatLoss])

MODEL=None
try:
    MODEL = DeepAR.load_from_checkpoint(
        MODEL_CKPT,
        target="log_return",
        loss=NormalDistributionLoss(),
        map_location="cpu"
    ).eval()
except Exception as e:
    print(f"[WARN] DeepAR load failed: {e}. Fallback to MA.")

def _build_infer_dataset(df_sym: pd.DataFrame):
    df=df_sym.sort_values("timestamp").copy()
    df["time_idx"]=np.arange(len(df), dtype=np.int64)
    return TimeSeriesDataSet(
        df, time_idx="time_idx", target="log_return", group_ids=["series_id"],
        max_encoder_length=SEQ_LEN, max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"], time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"], target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

class OnlineAdapter:
    def __init__(self, lr=0.05, maxbuf=240):
        self.a, self.b = 1.0, 0.0
        self.lr=float(lr); self.buf=collections.deque(maxlen=int(maxbuf))
        self.last_refit=0.0
    def transform(self, mu: float)->float:
        return self.a*mu + self.b
    def sgd_update(self, mu: float, y: float):
        pred=self.a*mu + self.b; e=pred - y
        self.a -= self.lr * e * mu
        self.b -= self.lr * e
        self.a = 0.999*self.a + 0.001*1.0
        self.b = 0.999*self.b
        self.buf.append((float(mu), float(y)))
    def maybe_refit(self, now_ts: float, period_sec: int = 120):
        if now_ts - self.last_refit < period_sec: return
        self.last_refit = now_ts
        if len(self.buf) < 20: return
        mu = np.array([m for m,_ in self.buf], dtype=float); y = np.array([yy for _,yy in self.buf], dtype=float)
        X = np.vstack([mu, np.ones_like(mu)]).T
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a_new, b_new = float(beta[0]), float(beta[1])
            if np.isfinite(a_new) and np.isfinite(b_new):
                self.a = float(np.clip(a_new, 0.2, 5.0))
                self.b = float(np.clip(b_new, -0.02, 0.02))
        except Exception:
            pass

_ADAPTERS: Dict[str, OnlineAdapter] = collections.defaultdict(lambda: OnlineAdapter(ONLINE_LR, ONLINE_MAXBUF))

def _fee_threshold():
    rt = 2.0*TAKER_FEE + 2.0*(SLIPPAGE_BPS_TAKER/1e4)
    return math.log(1.0 + FEE_SAFETY*rt)
_THR = _fee_threshold()

@torch.no_grad()
def _deepar_side(symbol: str):
    if MODEL is None: return None, 0.0
    df=_recent_1m_df(symbol, minutes=SEQ_LEN+PRED_LEN+10)
    if df is None or len(df) < (SEQ_LEN+1): return None, 0.0
    tsd=_build_infer_dataset(df)
    dl=tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
    mu=float(MODEL.predict(dl, mode="prediction")[0,0])
    mu_adj = _ADAPTERS[symbol].transform(mu) if MODEL_MODE=="online" else mu
    side=None
    if   mu_adj >  _THR: side="Buy"
    elif mu_adj < -_THR: side="Sell"
    if VERBOSE: print(f"[PRED] {symbol} mu={mu:.6g} mu'={mu_adj:.6g} thr={_THR:.6g} -> {side}")
    return (side, 1.0) if side else (None, 0.0)

def _ma_side(symbol:str)->Optional[str]:
    df=_recent_1m_df(symbol, minutes=120)
    if df is None or len(df)<60: return None
    c=df["close"].to_numpy()
    ma20, ma60 = np.mean(c[-20:]), np.mean(c[-60:])
    if ma20>ma60: return "Buy"
    if ma20<ma60: return "Sell"
    return None

def choose_entry(symbol:str):
    side, conf = _deepar_side(symbol)
    if side is None:
        side = _ma_side(symbol); conf = 0.6 if side else 0.0
    m=ENTRY_MODE
    if m=="inverse":
        if side=="Buy": side="Sell"
        elif side=="Sell": side="Buy"
    elif m=="random":
        side = side or ("Buy" if (time.time()*1000)%2<1 else "Sell")
    return side, float(conf or 0.0)

# ========= Qty / Orders =========
def _get_available_balance() -> float:
    try:
        wb = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows = (wb.get("result") or {}).get("list") or []
        if not rows: return 0.0
        c = next((x for x in rows[0].get("coin", []) if x.get("coin") == "USDT"), None)
        if not c: return 0.0
        bal = float(c.get("walletBalance") or 0.0) + float(c.get("unrealisedPnl") or 0.0)
        return max(bal, 0.0)
    except:
        return 0.0

def compute_qty(symbol: str, mid: float, equity: float) -> float:
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
    if mid <= 0: return 0.0
    avail = _get_available_balance()
    if avail <= 0: return 0.0
    target_im = avail * ENTRY_EQUITY_PCT * 0.9
    raw_qty = (target_im * LEVERAGE) / mid
    qty = math.floor(raw_qty / qty_step) * qty_step
    if qty < min_qty:
        qty = math.ceil(min_qty / qty_step) * qty_step
    return float(qty)

def place_market(symbol: str, side: str, qty: float) -> bool:
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
    q = math.floor(float(qty) / qty_step) * qty_step
    if q < min_qty:
        if VERBOSE: print(f"[SKIP] {symbol}: qty<{min_qty} (q={q})")
        return False
    def _fmt(qv: float) -> str:
        return (f"{qv:.10f}".rstrip("0").rstrip(".")) if not qty_step.is_integer() else str(int(round(qv)))
    if not _ORDER_SEM.acquire(timeout=2.0):
        return False
    try:
        attempts = 0
        while q >= min_qty and attempts < 5:
            try:
                _api_call("order", client.place_order,
                          category=CATEGORY, symbol=symbol, side=side,
                          orderType="Market", qty=_fmt(q), timeInForce="IOC", reduceOnly=False)
                return True
            except Exception as e:
                msg = str(e)
                if "110007" in msg or "not enough" in msg:
                    q = math.floor((q * 0.74) / qty_step) * qty_step
                    attempts += 1
                    continue
                if "10006" in msg or "Too many visits" in msg:
                    time.sleep(0.6); attempts += 1; continue
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
        if not pos: return
        pos_qty = float(pos.get("qty") or 0.0)
        if pos_qty <= 0: return
        target = min(float(qty), pos_qty)
        q = math.floor(target / step) * step
        if q < minq:
            if VERBOSE: print(f"[SKIP][close] {symbol}: q<{minq} (q={q}), pos_qty={pos_qty}")
            return
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

def place_reduce_limit(symbol: str, side: str, qty: float, price: float) -> bool:
    try:
        rule = get_instrument_rule(symbol)
        qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
        tick = float(rule["tickSize"])
        q = math.floor(qty / qty_step) * qty_step
        if q < min_qty:
            if VERBOSE: print(f"[SKIP][reduce] {symbol}: q<{min_qty} (q={q})"); return False
        px = math.floor(price / tick) * tick
        q_str = _fmt_to_step(q, qty_step)
        px_str = _fmt_to_step(px, tick)
        _api_call("order", client.place_order,
                  category=CATEGORY, symbol=symbol, side=side,
                  orderType="Limit", qty=q_str, price=px_str,
                  timeInForce="PostOnly", reduceOnly=True)
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
        return _api_call("order", client.set_trading_stop, **params)["retCode"] == 0
    except Exception as e:
        print(f"[ERR] set_trading_stop {symbol} -> {e}")
        return False

# ========= Timeout / Vol =========
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
TRADE_STATE:Dict[str,dict]={}

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
    side = pos["side"]; entry = float(pos.get("entry") or 0.0); qty = float(pos.get("qty") or 0.0)
    if entry<=0 or qty<=0: return False
    TRADE_STATE[symbol] = _build_trade_state(entry, qty, side)
    DEADLINE[symbol]    = time.time() + _calc_timeout_minutes(symbol)*60
    return True

# ========= Entry =========
def tp_from_bps(entry:float,bps:float,side:str)->float:
    return entry*(1+bps/10000.0) if side=="Buy" else entry*(1-bps/10000.0)

def attach_tp_sl(symbol: str, side: str, entry: float, qty: float, position_idx: Optional[int] = None):
    rule = get_instrument_rule(symbol)
    lot = float(rule["qtyStep"]); minq = float(rule["minOrderQty"])
    close_side = "Sell" if side == "Buy" else "Buy"
    def _rd(x, step):  return math.floor(x/step)*step
    tp1 = tp_from_bps(entry, TP1_BPS, side)
    tp2 = tp_from_bps(entry, TP2_BPS, side)
    tp3 = tp_from_bps(entry, TP3_BPS, side)
    sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    q1 = _rd(qty*TP1_RATIO, lot); q2 = _rd(qty*TP2_RATIO, lot)
    if q1 >= minq: place_reduce_limit(symbol, close_side, q1, tp1)
    if q2 >= minq: place_reduce_limit(symbol, close_side, q2, tp2)
    set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=position_idx)
    TRADE_STATE[symbol] = {"side": side, "entry": float(entry), "qty": float(qty),
                           "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
                           "tp1_filled": False, "tp2_filled": False, "be_moved": False}

def try_enter(symbol:str):
    now = time.time()
    if COOLDOWN_UNTIL.get(symbol,0.0) > now: return
    pos_map = get_positions()
    if symbol in pos_map: return
    if len(pos_map) >= MAX_OPEN: return
    if _BUCKET_ORDER.tokens < 0.9: return

    # === 모델 신호 ===
    side, conf = choose_entry(symbol)
    if side not in ("Buy","Sell"):
        if VERBOSE: print(f"[SKIP][SIGNAL] {symbol} side=None"); return

    bid, ask, mid = get_quote(symbol)
    if mid <= 0: return

    eq = get_wallet_equity()
    qty = compute_qty(symbol, mid, eq)
    if qty <= 0: return

    ok = place_market(symbol, side, qty)
    if not ok: return

    pos = get_positions().get(symbol)
    if not pos: return
    entry = float(pos.get("entry") or 0.0) or mid
    pidx = pos.get("positionIdx") if isinstance(pos, dict) else None

    attach_tp_sl(symbol, side, entry, float(pos.get("qty") or qty), position_idx=pidx)
    DEADLINE[symbol] = time.time() + _calc_timeout_minutes(symbol) * 60

    if not init_trade_state_from_positions(symbol):
        horizon_min = _calc_timeout_minutes(symbol)
        DEADLINE[symbol] = now + horizon_min * 60
        TRADE_STATE[symbol] = _build_trade_state(mid, qty, side)

    _log_trade({
        "ts": int(now), "event": "ENTRY", "symbol": symbol, "side": side,
        "qty": f"{qty:.8f}", "entry": f"{mid:.6f}", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
        "reason": "open", "mode": ENTRY_MODE
    })
    if VERBOSE: print(f"[OPEN] {symbol} {side} {qty}@{mid:.4f}")

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
                side = p["side"]; entry = float(p.get("entry") or 0.0); qty = float(p.get("qty") or 0.0)
                if entry <= 0 or qty <= 0: continue
                attach_tp_sl(sym, side, entry, qty, position_idx=p.get("positionIdx"))
                DEADLINE[sym] = time.time() + _calc_timeout_minutes(sym) * 60
                if sym not in TRADE_STATE:
                    init_trade_state_from_positions(sym)

            # 온라인 어댑터 업데이트(실제 y와 μ근사)
            if MODEL_MODE == "online":
                for sym in pos.keys():
                    df = _recent_1m_df(sym, minutes=3)
                    if df is not None and len(df) >= 2:
                        y = float(np.log(df["close"].iloc[-1] / max(df["close"].iloc[-2], 1e-12)))
                        st = TRADE_STATE.get(sym)
                        if st:
                            _,_,mark = get_quote(sym)
                            mu_est = float(np.clip(np.log(max(mark,1e-12)/max(st["entry"],1e-12)), -0.01, 0.01))
                        else:
                            mu_est = 0.0
                        _ADAPTERS[sym].sgd_update(mu_est, y)
                        _ADAPTERS[sym].maybe_refit(time.time(), ADAPTER_REFIT_SEC)

            # 포지션 관리
            for sym, st in list(TRADE_STATE.items()):
                p = pos.get(sym)
                if not p:
                    COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                    TRADE_STATE.pop(sym,None); DEADLINE.pop(sym,None); continue

                side=st["side"]; entry=st["entry"]; qty=st["qty"]
                tp1, tp2, tp3, sl = st["tp1"], st["tp2"], st["tp3"], st["sl"]
                _,_,mark=get_quote(sym)
                if mark<=0: continue

                rule = get_instrument_rule(sym)
                step = float(rule["qtyStep"]); minq = float(rule["minOrderQty"])

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
                    part = math.floor((st["qty"] * TP2_RATIO) / step) * step
                    if part < minq: part = math.floor(st["qty"] / step) * step
                    if part >= minq and part <= st["qty"]:
                        close_market(sym, side, part)
                        st["tp2_filled"] = True
                        st["qty"] = qty = max(0.0, st["qty"] - part)
                        _log_trade({"ts": int(time.time()), "event": "TP2_FILL", "symbol": sym, "side": side,
                                    "qty": f"{part:.8f}", "entry": f"{entry:.6f}", "exit": f"{tp2:.6f}",
                                    "tp1": "", "tp2": "", "tp3": "", "sl": "", "reason": "", "mode": "paper"})

                # BE 이동
                if st["tp2_filled"] and not st["be_moved"]:
                    st["sl"]=entry*(1+BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1-BE_EPS_BPS/10000.0)
                    st["be_moved"]=True
                    _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,"qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":f"{st['sl']:.6f}","reason":"BE","mode":"paper"})

                # Trailing
                if (st["tp1_filled"] if TRAIL_AFTER_TIER==1 else st["tp2_filled"]):
                    if side=="Buy": st["sl"]=max(st["sl"], mark*(1-TRAIL_BPS/10000.0))
                    else: st["sl"]=min(st["sl"], mark*(1+TRAIL_BPS/10000.0))

                # Exit 조건
                pnl=(mark-entry)*qty if side=="Buy" else (entry-mark)*qty
                hit_abs=pnl>=ABS_TP_USD
                hit_tp3=(side=="Buy" and mark>=tp3) or (side=="Sell" and mark<=tp3)
                hit_sl=(side=="Buy" and mark<=st["sl"]) or (side=="Sell" and mark>=st["sl"])
                hit_to=(now>=DEADLINE.get(sym,now+1e9))

                if hit_abs or hit_tp3 or hit_sl or hit_to:
                    close_market(sym,side,st["qty"])
                    _log_trade({"ts":int(time.time()),"event":"EXIT","symbol":sym,"side":side,"qty":f"{st['qty']:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}",
                                "tp1":"","tp2":"","tp3":"","sl":"","reason":"ABS_TP" if hit_abs else ("TP3" if hit_tp3 else ("SL" if hit_sl else "HORIZON_TIMEOUT")),"mode":"paper"})
                    COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC; TRADE_STATE.pop(sym,None); DEADLINE.pop(sym,None)

            time.sleep(1)
        except Exception as e:
            print("[MONITOR ERR]",e); time.sleep(2)

# ========= Scanner =========
def scanner_loop():
    idx = 0
    while True:
        try:
            pos = get_positions()
            free_slots = max(0, MAX_OPEN - len(pos))
            if free_slots <= 0:
                time.sleep(1); continue
            batch = SYMBOLS[idx:idx+free_slots]
            if not batch:
                idx = 0; batch = SYMBOLS[:free_slots]
            idx = (idx + free_slots) % len(SYMBOLS)
            if SCAN_PARALLEL and free_slots > 1:
                with ThreadPoolExecutor(max_workers=min(SCAN_WORKERS, free_slots)) as ex:
                    ex.map(try_enter, batch)
            else:
                for s in batch: try_enter(s)
            time.sleep(1.5)
        except Exception as e:
            print("[SCANNER ERR]", e); time.sleep(2)

# ========= Main =========
def main():
    print(f"[START] PAPER-Equivalent LIVE EXT | MODE={ENTRY_MODE} MODEL_MODE={MODEL_MODE} TESTNET={TESTNET}")
    t1=threading.Thread(target=monitor_loop,daemon=True); t2=threading.Thread(target=scanner_loop,daemon=True)
    t1.start(); t2.start()
    while True: time.sleep(60)

if __name__=="__main__":
    main()


# =======================
# === DUAL-DEEPAR PATCH (appended) ===
# Two-model agreement: same sign AND (loose: either |mu|>=thr, strict: both).
# Env:
#   MODEL_CKPT_MAIN, MODEL_CKPT_ALT
#   THR_MODE=fixed|fee   (default fixed)
#   DUAL_RULE=loose|strict (default loose)
#   MODEL_THR_BPS=5      (used when THR_MODE=fixed)
# =======================
try:
    _VERBOSE_FLAG = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")
except Exception:
    _VERBOSE_FLAG = True

# Threshold resolver
def _resolve_thr():
    mode = os.getenv("THR_MODE", "fixed").lower()
    if mode == "fee":
        # Prefer existing fee_threshold if present, otherwise fallback to fixed 5bps
        try:
            return float(fee_threshold())
        except Exception:
            pass
        bps = float(os.getenv("MODEL_THR_BPS","5.0"))
        return bps / 1e4
    else:
        bps = float(os.getenv("MODEL_THR_BPS","5.0"))
        return bps / 1e4

# Optional adapters
def _maybe_adapt(symbol, mu):
    try:
        # try dedicated adapters if available
        if 'MODEL_MODE' in globals() and MODEL_MODE == "online":
            if '_ADAPTERS_MAIN' in globals() and symbol in _ADAPTERS_MAIN:
                return _ADAPTERS_MAIN[symbol].transform(mu)
            if '_ADAPTERS_ALT' in globals() and symbol in _ADAPTERS_ALT:
                return _ADAPTERS_ALT[symbol].transform(mu)
            if '_ADAPTERS' in globals() and symbol in _ADAPTERS:
                return _ADAPTERS[symbol].transform(mu)
    except Exception:
        pass
    return mu

# Dual checkpoints
MODEL_CKPT_MAIN = os.getenv("MODEL_CKPT_MAIN", "models/multi_deepar_main.ckpt")
MODEL_CKPT_ALT  = os.getenv("MODEL_CKPT_ALT",  "models/multi_deepar_alt.ckpt")

DEEPar_MAIN = globals().get('DEEPar_MAIN', None)
DEEPar_ALT  = globals().get('DEEPar_ALT',  None)

try:
    if 'DeepAR' in globals():
        if DEEPar_MAIN is None:
            DEEPar_MAIN = DeepAR.load_from_checkpoint(MODEL_CKPT_MAIN, map_location="cpu").eval()
        if DEEPar_ALT is None:
            DEEPar_ALT  = DeepAR.load_from_checkpoint(MODEL_CKPT_ALT,  map_location="cpu").eval()
except Exception as _e:
    if _VERBOSE_FLAG:
        print(f"[WARN] Dual DeepAR load failed: {_e}")

import torch
@torch.no_grad()
def dual_deepar_direction(symbol: str):
    # Require both dual models
    if (globals().get('DEEPar_MAIN', None) is None) or (globals().get('DEEPar_ALT', None) is None):
        # fallback to single-model direction if available
        if 'deepar_direction' in globals():
            return deepar_direction(symbol)
        return (None, 0.0)

    # Fetch recent data using the same helpers as the original file
    _SEQ_LEN  = int(os.getenv("SEQ_LEN","240"))
    _PRED_LEN = int(os.getenv("PRED_LEN","60"))
    # The file may define either get_recent_1m or a similar function
    if 'get_recent_1m' not in globals():
        return (None, 0.0)
    df = get_recent_1m(symbol, minutes=_SEQ_LEN + _PRED_LEN + 10)
    if df is None or len(df) < (_SEQ_LEN + 1):
        return (None, 0.0)

    # Build dataset with existing builder
    builder_name = None
    for name in ('build_infer_dataset', '_build_infer_dataset'):
        if name in globals():
            builder_name = name
            break
    if builder_name is None:
        return (None, 0.0)
    tsd = globals()[builder_name](df)
    dl  = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)

    # Predict
    mu1 = float(DEEPar_MAIN.predict(dl, mode="prediction")[0, 0])
    mu2 = float(DEEPar_ALT.predict(dl,  mode="prediction")[0, 0])

    mu1a = _maybe_adapt(symbol, mu1)
    mu2a = _maybe_adapt(symbol, mu2)

    thr = _resolve_thr()
    rule = os.getenv("DUAL_RULE","loose").lower()

    if _VERBOSE_FLAG:
        try:
            print(f"[DEBUG2] {symbol} mu1={mu1:.6g}->{mu1a:.6g}  mu2={mu2:.6g}->{mu2a:.6g}  thr={thr:.6g} rule={rule}")
        except Exception:
            pass

    same_sign = (mu1a > 0 and mu2a > 0) or (mu1a < 0 and mu2a < 0)
    mag1 = abs(mu1a) >= thr
    mag2 = abs(mu2a) >= thr

    base = None
    if same_sign:
        if (rule == "strict" and (mag1 and mag2)) or (rule != "strict" and (mag1 or mag2)):
            base = "Buy" if (mu1a + mu2a) > 0 else "Sell"

    if base is None:
        return (None, 0.0)

    # Apply ENTRY_MODE if defined
    side = base
    try:
        m = (ENTRY_MODE or "model").lower()
        if m == "inverse":
            side = "Sell" if base == "Buy" else ("Buy" if base == "Sell" else None)
        elif m == "random":
            import random, time as _t
            side = side or (random.choice(["Buy","Sell"]))
    except Exception:
        pass

    return side, (1.0 if side is not None else 0.0)

# Override choose_entry to prefer dual first
def choose_entry(symbol: str):
    # dual first
    side, conf = dual_deepar_direction(symbol)
    if side in ("Buy","Sell"):
        return side, float(conf or 0.0)

    # fallback to existing single-model logic if available
    if 'deepar_direction' in globals():
        side, conf = deepar_direction(symbol)
    else:
        side, conf = (None, 0.0)

    # apply ENTRY_MODE consistently
    try:
        m = (ENTRY_MODE or "model").lower()
        if m == "inverse":
            if side == "Buy": side = "Sell"
            elif side == "Sell": side = "Buy"
        elif m == "random":
            import random
            side = side or random.choice(["Buy","Sell"])
    except Exception:
        pass

    return side, float(conf or 0.0)
# === END OF DUAL-DEEPAR PATCH ===
