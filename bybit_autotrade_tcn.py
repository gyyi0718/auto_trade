# -*- coding: utf-8 -*-
"""
Bybit Futures Live Trading (Unified v5, Linear Perp)
- Dual-Model gating: DeepAR + TCN -> same direction, and |TCN_mu'| >= thr_tcn
- PostOnly(LIMIT PostOnly) / Market entry
- TP1/TP2 reduce-only LIMIT, TP3/SL via set_trading_stop (closePosition)
- BE move + trailing, ATR timeout, cooldown
"""

import os, time, math, csv, threading, logging, random, collections, re
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from logging.handlers import RotatingFileHandler
from pytorch_forecasting.metrics import NormalDistributionLoss
from numpy.core.multiarray import _reconstruct as _np_reconstruct

# ===== Safe Globals for ckpt =====
class SignAwareNormalLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, *args, **kwargs): return torch.tensor(0.0)
from pytorch_forecasting.data.encoders import GroupNormalizer
add_safe_globals([GroupNormalizer, SignAwareNormalLoss])
add_safe_globals([np.ndarray, _np_reconstruct, SignAwareNormalLoss])
# ===== Logger =====
LOGGER = logging.getLogger("live_trade")
def setup_live_logger(log_file="live_trade.log", level=logging.INFO):
    LOGGER.setLevel(level); LOGGER.propagate=False
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    for h in list(LOGGER.handlers): LOGGER.removeHandler(h)
    ch=logging.StreamHandler(); ch.setLevel(level); ch.setFormatter(fmt)
    fh=RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level); fh.setFormatter(fmt)
    LOGGER.addHandler(ch); LOGGER.addHandler(fh)
    LOGGER.info("=== Logger initialized ===")

_GLOBAL_BAN_UNTIL = 0.0
_KLINE_CACHE = {}
_QUOTE_CACHE = {}
_CACHE_TTL_KLINE = float(os.getenv("CACHE_TTL_KLINE", "5"))
_CACHE_TTL_QUOTE = float(os.getenv("CACHE_TTL_QUOTE", "1.0"))
_BACKOFF_ON_1003 = float(os.getenv("BACKOFF_ON_1003", "60"))

# ===== ENV =====
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")                 # model|inverse|random
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "maker").lower() # maker|taker
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower() # maker|taker
MODEL_MODE = os.getenv("MODEL_MODE", "online").lower()        # fixed|online

LEVERAGE   = float(os.getenv("LEVERAGE", "5"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY", "1.20"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))
ENTRY_NOTIONAL_USD = float(os.getenv("ENTRY_NOTIONAL_USD", "25"))  # 예: 25 USDT

MIN_CONF = float(os.getenv("MIN_CONF", "0.30"))
TCN_THR_BPS = float(os.getenv("TCN_THR_BPS", "3"))
AGREE_MODE = os.getenv("AGREE_MODE", "soft")                  # strict|soft|tcn_only

ENTRY_EQUITY_PCT   = float(os.getenv("ENTRY_EQUITY_PCT", "0.50"))
MAX_OPEN   = int(os.getenv("MAX_OPEN", "5"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "10"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")

# TP/SL
TP1_BPS, TP2_BPS, TP3_BPS = float(os.getenv("TP1_BPS","50")), float(os.getenv("TP2_BPS","100")), float(os.getenv("TP3_BPS","200"))
TP1_RATIO, TP2_RATIO = float(os.getenv("TP1_RATIO","0.40")), float(os.getenv("TP2_RATIO","0.35"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
HARD_SL_PCT = float(os.getenv("HARD_SL_PCT","0.02"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))

# Timeout (ATR)
TIMEOUT_MODE = os.getenv("TIMEOUT_MODE", "atr")
VOL_WIN = int(os.getenv("VOL_WIN", "60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K", "1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "20"))
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))

# Universe / IO
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,LINEAUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT").split(",")

TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]

SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","0")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","1"))

# ===== Bybit Unified v5 =====
from pybit.unified_trading import HTTP
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY","Dlp4eJD6YFmO99T8vC")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET","YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U")
CATEGORY = "linear"

bybit = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)

def _handle_bybit_exc(e: Exception):
    global _GLOBAL_BAN_UNTIL
    msg = str(e)
    # -1003 대응 유사 처리(메시지 기반 백오프)
    if "1003" in msg or "Too many visits" in msg or "rate" in msg.lower():
        now = time.time()
        _GLOBAL_BAN_UNTIL = max(_GLOBAL_BAN_UNTIL, now + _BACKOFF_ON_1003)
        wait = max(0.0, _GLOBAL_BAN_UNTIL - now)
        LOGGER.warning(f"[RATE][BAN] detected. Backing off for ~{wait:.0f}s")

# ===== CSV IO =====
def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(EQUITY_FIELDS)

def _log_trade(row:dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])

def _log_equity(eq:float):
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{eq:.6f}"])

# ===== Utils =====
def _round_down(x: float, step: float) -> float:
    if step<=0: return x
    return math.floor(x/step)*step

def _round_to_tick(x: float, tick: float) -> float:
    if tick<=0: return x
    return math.floor(x/tick)*tick

def _fmt_qty(q: float, step: float) -> str:
    s=str(step); dec=0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{q:.{dec}f}" if dec>0 else str(int(round(q)))).rstrip("0").rstrip(".")

def _fmt_px(p: float, tick: float) -> str:
    s=str(tick); dec=0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{p:.{dec}f}" if dec>0 else str(int(round(p)))).rstrip("0").rstrip(".")

# ===== Bybit helpers =====
_RULE_CACHE={}
def get_instrument_rule(symbol:str)->dict:
    try:
        if symbol in _RULE_CACHE: return _RULE_CACHE[symbol]
        r = bybit.get_instruments_info(category=CATEGORY, symbol=symbol)
        item = ((r.get("result") or {}).get("list") or [{}])[0]
        pf = (item.get("priceFilter") or {})
        lf = (item.get("lotSizeFilter") or {})
        rule = {
            "tickSize": float(pf.get("tickSize") or 0.01),
            "qtyStep":  float(lf.get("qtyStep") or 0.001),
            "minOrderQty": float(lf.get("minOrderQty") or 0.001),
            "minNotional": 5.0
        }
        _RULE_CACHE[symbol]=rule
        return rule
    except Exception as e:
        _handle_bybit_exc(e)
        return {"tickSize":0.01,"qtyStep":0.001,"minOrderQty":0.001,"minNotional":5.0}

def get_wallet_equity() -> float:
    """
    Bybit UTA: coin[].walletBalance가 0이어도 account 레벨 totalEquity가 유효.
    두 값 중 큰 값을 사용.
    """
    try:
        r = bybit.get_wallet_balance(accountType="UNIFIED")
        acc = ((r.get("result") or {}).get("list") or [{}])[0]
        total_eq = float(acc.get("totalEquity") or 0.0)
        coins = acc.get("coin") or []
        usdt = next((c for c in coins if c.get("coin") == "USDT"), None)
        usdt_bal = float(usdt.get("walletBalance")) if usdt else 0.0
        return max(total_eq, usdt_bal)
    except Exception as e:
        _handle_bybit_exc(e)
        return 0.0


def get_positions()->Dict[str,dict]:
    try:
        r = bybit.get_positions(category=CATEGORY)
        rows = (r.get("result") or {}).get("list") or []
    except Exception as e:
        _handle_bybit_exc(e); rows=[]
    out={}
    for p in rows:
        sym=p.get("symbol")
        sz = float(p.get("size") or 0.0)
        if abs(sz) <= 0:
            continue
        out[sym]={
            "side": p.get("side"),                         # "Buy" | "Sell"
            "qty": abs(sz),
            "entry": float(p.get("avgPrice") or 0.0),
            "positionIdx": int(p.get("positionIdx") or 0),
            "tp": float(p.get("takeProfit") or 0.0),       # ← 추가
            "sl": float(p.get("stopLoss")   or 0.0),       # ← 추가
        }
    return out

def ensure_isolated_and_leverage(symbol: str):
    try:
        bybit.switch_position_mode(category=CATEGORY, mode=0)  # one-way
    except Exception: pass
    try:
        bybit.set_margin_mode(category=CATEGORY, symbol=symbol, tradeMode=1)  # 1=Isolated, 0=Cross
    except Exception: pass
    try:
        lv=int(max(1, min(int(LEVERAGE), 100)))
        bybit.set_leverage(category=CATEGORY, symbol=symbol, buyLeverage=str(lv), sellLeverage=str(lv))
    except Exception: pass

def get_quote(symbol:str)->Tuple[float,float,float]:
    now=time.time()
    if now < _GLOBAL_BAN_UNTIL:
        c=_QUOTE_CACHE.get(symbol)
        return c[1] if c and (now - c[0] <= _CACHE_TTL_QUOTE) else (0.0,0.0,0.0)
    c=_QUOTE_CACHE.get(symbol)
    if c and (now - c[0] <= _CACHE_TTL_QUOTE): return c[1]
    try:
        r = bybit.get_tickers(category=CATEGORY, symbol=symbol)
        row = ((r.get("result") or {}).get("list") or [{}])[0]
        bid = float(row.get("bid1Price") or 0.0)
        ask = float(row.get("ask1Price") or 0.0)
        last = float(row.get("lastPrice") or 0.0)
        mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask or 0.0)
        _QUOTE_CACHE[symbol]=(now,(bid,ask,mid))
        return bid, ask, mid
    except Exception as e:
        _handle_bybit_exc(e)
        c=_QUOTE_CACHE.get(symbol)
        return c[1] if c else (0.0,0.0,0.0)

def _recent_1m_df(symbol: str, minutes: int = 200):
    now=time.time()
    limit=min(max(minutes,1), 1000)
    if now < _GLOBAL_BAN_UNTIL:
        c=_KLINE_CACHE.get(symbol); return c[1] if c else None
    c=_KLINE_CACHE.get(symbol)
    if c and (now - c[0] <= _CACHE_TTL_KLINE): return c[1]
    try:
        r = bybit.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=limit)
        rows = (r.get("result") or {}).get("list") or []
        if not rows: return c[1] if c else None
        rows = rows[::-1]
        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
            "series_id": symbol,
            "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
            "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
        } for z in rows])
        df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
        _KLINE_CACHE[symbol]=(now, df)
        return df
    except Exception as e:
        _handle_bybit_exc(e)
        c=_KLINE_CACHE.get(symbol); return c[1] if c else None

# fees / filters
def _entry_cost_bps()->float:
    fee = (0.0 if EXECUTION_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXECUTION_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))
def _exit_cost_bps()->float:
    fee = (0.0 if EXIT_EXEC_MODE=="maker" else TAKER_FEE) * 10_000.0
    slip= (SLIPPAGE_BPS_MAKER if EXIT_EXEC_MODE=="maker" else SLIPPAGE_BPS_TAKER)
    return max(0.0, fee + max(0.0, slip))
def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0
def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)
def compute_sl(entry:float, side:str)->float:
    dyn = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    hard= entry*(1.0 - HARD_SL_PCT) if side=="Buy" else entry*(1.0 + HARD_SL_PCT)
    return max(dyn, hard) if side=="Buy" else min(dyn, hard)

def _get_available_balance():
    try:
        r = bybit.get_wallet_balance(accountType="UNIFIED")
        acc = ((r.get("result") or {}).get("list") or [{}])[0]
        # Bybit UTA: totalAvailableBalance 가 주문 가능 기준
        return float(acc.get("totalAvailableBalance") or 0.0)
    except Exception:
        return 0.0

def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float) -> Tuple[float, float]:
    rule = get_instrument_rule(symbol)
    qty_step = float(rule["qtyStep"]); min_qty = float(rule["minOrderQty"])
    if mid <= 0: return 0.0, 0.0
    lv = max(1.0, float(LEVERAGE))

    avail = _get_available_balance()
    # 버퍼 90% 적용해 과대 주문 방지
    cap_notional = max(avail * lv * 0.90, 0.0)
    # 사용자가 지정한 기본 명목가
    want_notional = 0.0
    if equity_now > 0 and pct > 0:
        want_notional = equity_now * pct * lv
    want_notional = max(want_notional, ENTRY_NOTIONAL_USD)

    target_notional = max(10.0, min(want_notional, cap_notional))  # ≥$10, ≤가용한도
    qty = target_notional / mid
    qty = math.ceil(qty / qty_step) * qty_step
    if qty < min_qty:
        qty = math.ceil(min_qty / qty_step) * qty_step
    return qty, qty * mid

# ===== Orders (Bybit) =====
def _order_with_mode_retry(params: dict):
    try:
        r = bybit.place_order(**params)
        if (r or {}).get("retCode") == 0:
            return (r.get("result") or {}).get("orderId")
        # 110007이면 qty 절반으로 1회 재시도
        if str((r or {}).get("retCode")) == "110007" and "qty" in params:
            try:
                q = float(params["qty"])
                if q > 0:
                    params["qty"] = str(max(q/2, 1e-12))
                    r2 = bybit.place_order(**params)
                    if (r2 or {}).get("retCode") == 0:
                        return (r2.get("result") or {}).get("orderId")
            except Exception:
                pass
        LOGGER.warning(f"[ORDER][REJECT] {params.get('symbol')} -> {r}")
        return None
    except Exception as e:
        LOGGER.warning(f"[ORDER][ERR] {params.get('symbol')} -> {e} | params={params}")
        _handle_bybit_exc(e)
        return None


def place_market(symbol: str, side: str, qty: float) -> bool:
    rule = get_instrument_rule(symbol)
    q = _round_down(qty, rule["qtyStep"])
    if q < rule["minOrderQty"]:
        return False
    params = dict(
        category=CATEGORY, symbol=symbol, side=side,
        orderType="Market", qty=_fmt_qty(q, rule["qtyStep"]),
        reduceOnly=False
    )
    return _order_with_mode_retry(params) is not None

def place_limit_postonly(symbol: str, side: str, qty: float, price: float):
    rule = get_instrument_rule(symbol)
    q  = _round_down(qty, rule["qtyStep"])
    px = _round_to_tick(price, rule["tickSize"])
    if q < rule["minOrderQty"]:
        LOGGER.info(f"[POSTONLY][SKIP] {symbol} q<{rule['minOrderQty']} q={q}")
        return None
    params = dict(
        category=CATEGORY, symbol=symbol, side=side,
        orderType="Limit", qty=_fmt_qty(q, rule["qtyStep"]),
        price=_fmt_px(px, rule["tickSize"]),
        timeInForce="PostOnly", reduceOnly=False
    )
    LOGGER.info(f"[POSTONLY][TRY] {symbol} side={side} qty={q} px={px}")
    oid = _order_with_mode_retry(params)
    if oid is None:
        LOGGER.warning(f"[POSTONLY][REJECT] {symbol} side={side} qty={q} px={px}")
    return oid

def place_reduce_limit(symbol: str, close_side: str, qty: float, price: float) -> bool:
    rule = get_instrument_rule(symbol)
    q  = _round_down(qty, rule["qtyStep"])
    px = _round_to_tick(price, rule["tickSize"])
    if q < rule["minOrderQty"]: return False
    params = dict(
        category=CATEGORY, symbol=symbol, side=close_side,
        orderType="Limit", qty=_fmt_qty(q, rule["qtyStep"]),
        price=_fmt_px(px, rule["tickSize"]),
        timeInForce="PostOnly",  # 또는 "GTC"도 OK. PostOnly를 선호하면 그대로.
        reduceOnly=True
    )
    return _order_with_mode_retry(params) is not None

def set_trading_stop(symbol: str,
                     side: str,
                     entry: float,
                     sl_price: float | None = None,
                     tp_price: float | None = None,
                     pos_idx: int | None = None,
                     mode: str = "Full") -> bool:
    rule = get_instrument_rule(symbol)
    tick = float(rule["tickSize"])

    def _round_ok(p):
        if p is None or p <= 0:
            return None
        p = _round_to_tick(p, tick)
        # 엔트리와 동일값 방지 위해 한 틱 바깥으로 민다
        if abs(p - entry) < tick:
            p = entry - tick if side=="Buy" else entry + tick
        return float(_fmt_px(p, rule["tickSize"]))

    tp = _round_ok(tp_price)
    sl = _round_ok(sl_price)

    if side == "Buy":
        if tp is not None and tp <= entry: tp = None
        if sl is not None and sl >= entry: sl = None
    else:
        if tp is not None and tp >= entry: tp = None
        if sl is not None and sl <= entry: sl = None

    if pos_idx is None:
        pos = get_positions().get(symbol)
        if not pos:
            LOGGER.warning(f"[STOP][MISS] {symbol} no position")
            return False
        pos_idx = int(pos.get("positionIdx") or 0)

    if tp is None and sl is None:
        LOGGER.info(f"[STOP][SKIP] {symbol} invalid tp/sl after checks")
        return False

    try:
        params = dict(
            category=CATEGORY,
            symbol=symbol,
            positionIdx=int(pos_idx),       # one-way=0
            tpslMode=mode,                  # "Full" | "Partial"
            tpTriggerBy="MarkPrice",
            slTriggerBy="MarkPrice",
            tpOrderType="Market",           # ← 명시
            slOrderType="Market",           # ← 명시
        )
        if tp is not None: params["takeProfit"] = str(tp)
        if sl is not None: params["stopLoss"]   = str(sl)

        r = bybit.set_trading_stop(**params)
        ok = (r or {}).get("retCode") == 0
        if ok:
            LOGGER.info(f"[STOP][OK] {symbol} tp={tp} sl={sl} posIdx={pos_idx} mode={mode} resp={(r or {}).get('retCode')}")
        else:
            LOGGER.warning(f"[STOP][FAIL] {symbol} resp={r}")
        return ok
    except Exception as e:
        LOGGER.warning(f"[STOP][ERR] {symbol} -> {e}")
        return False

def set_stop_with_retry(symbol: str, sl_price: float | None = None, tp_price: float | None = None, position_idx: int | None = None) -> bool:
    rule = get_instrument_rule(symbol); tick = float(rule.get("tickSize") or 0.0)
    def _rt(x):
        return None if (x is None or x <= 0) else _round_to_tick(float(x), tick)
    sl = _rt(sl_price); tp = _rt(tp_price)
    if sl is None and tp is None:
        LOGGER.info(f"[TRADING_STOP][SKIP] {symbol} empty tp/sl"); return False

    # 포지션 인덱스 항상 명시 (One-way=0, Hedge: Buy=1 / Sell=2)
    if position_idx is None:
        pos = get_positions().get(symbol)
        if not pos:
            LOGGER.warning(f"[TRADING_STOP][MISS] {symbol} no position"); return False
        position_idx = int(pos.get("positionIdx") or 0)

    params = dict(
        category=CATEGORY, symbol=symbol, positionIdx=position_idx,
        tpslMode="Full", tpTriggerBy="MarkPrice", slTriggerBy="MarkPrice"
    )
    if tp is not None:
        params["takeProfit"] = _fmt_px(tp, tick)
        params["tpOrderType"] = "Market"
    if sl is not None:
        params["stopLoss"] = _fmt_px(sl, tick)
        params["slOrderType"] = "Market"

    try:
        LOGGER.info(f"[TRADING_STOP][REQ] {symbol} tp={tp} sl={sl} posIdx={position_idx}")
        r = bybit.set_trading_stop(**params)
        ok = (r or {}).get("retCode") == 0
        LOGGER.info(f"[TRADING_STOP][ACK] {symbol} ok={ok} resp={r}")
        return ok
    except Exception as e:
        LOGGER.warning(f"[TRADING_STOP][ERR] {symbol} -> {e}")
        return False


def _ensure_tpsl(symbol: str):
    """포지션 존재 시 TP3/SL 재확인 및 보정."""
    pos = get_positions().get(symbol)
    if not pos: return
    side = pos["side"]; entry = float(pos.get("entry") or 0.0)
    if entry <= 0: return

    # 목표 재계산
    tp3 = tp_from_bps(entry, TP3_BPS, side)
    sl  = compute_sl(entry, side)


    # 변경:
    set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=int(pos.get("positionIdx") or 0))



def get_open_orders(symbol: Optional[str]=None, order_filter: Optional[str]=None) -> List[dict]:
    """
    order_filter: None | "Order" | "Stop"
    - None  -> 기본(서버 기본값)
    - Order -> 일반 지정/시장 등
    - Stop  -> TP/SL/트리거 주문
    """
    try:
        params = dict(category=CATEGORY)
        if symbol: params["symbol"] = symbol
        if order_filter in ("Order","Stop"):
            params["orderFilter"] = order_filter
        r = bybit.get_open_orders(**params)
        return ((r.get("result") or {}).get("list") or [])
    except Exception:
        return []

def debug_open_state(symbol: str):
    ords  = get_open_orders(symbol, order_filter="Order")
    stops = get_open_orders(symbol, order_filter="Stop")
    pos   = get_positions().get(symbol)

    LOGGER.info(f"[DBG][OPEN] {symbol} normal={len(ords)} stop={len(stops)} pos={'Y' if pos else 'N'}")
    for x in ords:
        LOGGER.info(f"[DBG][ORD] id={x.get('orderId')} side={x.get('side')} "
                    f"qty={x.get('qty')} px={x.get('price')} reduceOnly={x.get('reduceOnly')}")
    for x in stops:
        LOGGER.info(f"[DBG][STP] id={x.get('orderId')} side={x.get('side')} "
                    f"type={x.get('orderType')} trigger={x.get('triggerPrice')} ")
    if pos:
        LOGGER.info(f"[DBG][POS] side={pos['side']} qty={pos['qty']} entry={pos['entry']} "
                    f"tp={pos.get('tp',0.0)} sl={pos.get('sl',0.0)} posIdx={pos.get('positionIdx')}")


def _cancel_order(symbol:str, order_id:str)->bool:
    try:
        r = bybit.cancel_order(category=CATEGORY, symbol=symbol, orderId=order_id)
        return (r or {}).get("retCode")==0
    except Exception as e:
        LOGGER.warning(f"[CANCEL ERR] {symbol} {order_id}: {e}")
        return False

# ===== Vol / timeout =====
def _recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    k=_recent_1m_df(symbol, minutes)
    if k is None or len(k)<10: return 10.0
    lr=np.diff(np.log(k["close"].to_numpy()))
    v_bps=float(np.median(np.abs(lr))*1e4)
    return max(v_bps,1.0)

def _calc_timeout_minutes(symbol: str) -> int:
    if TIMEOUT_MODE=="fixed":
        return max(1, int(math.ceil(max(1, MAX_HOLD_SEC)/60.0)))
    target_bps = float(os.getenv("TIMEOUT_TARGET_BPS") or TP1_BPS)
    v_bps = _recent_vol_bps(symbol, VOL_WIN)
    need = int(math.ceil((target_bps / v_bps) * TIMEOUT_K))
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

# ===== Models (DeepAR + TCN) =====
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS","1"))))
DEVICE = torch.device("cpu")

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR

SEQ_LEN = int(os.getenv("SEQ_LEN","120"))
PRED_LEN = int(os.getenv("PRED_LEN","30"))

DEEPar=None
DEEPar_CKPT = os.getenv("MODEL_CKPT_DEEPAR","D:/ygy_work/coin/multimodel/models/multi_deepar_best_main.ckpt")
try:
    # safe globals 등록 후 로드
    from pytorch_forecasting.models.deepar import DeepAR
    from pytorch_forecasting.metrics import NormalDistributionLoss
    DEEPar = DeepAR.load_from_checkpoint(DEEPar_CKPT, map_location="cpu")
    # ckpt 안의 loss 심벌 무시하고 런타임 Loss로 교체
    DEEPar.loss = NormalDistributionLoss()
    DEEPar.eval().to("cpu")
except Exception as e:
    LOGGER.warning(f"[DEEPar][LOAD][WARN] {e}")
def _build_infer_dataset(df_sym: pd.DataFrame):
    df = df_sym.sort_values("timestamp").copy()
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

class SimpleTCN(torch.nn.Module):
    def __init__(self, in_ch=1, hid=64, ks=3, levels=4):
        super().__init__()
        layers=[]; ch=in_ch
        for i in range(levels):
            layers.append(torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(ch, hid, ks, padding=ks//2)
            ))
            layers.append(torch.nn.ReLU())
            ch=hid
        self.tcn=torch.nn.Sequential(*layers)
        self.head=torch.nn.Linear(hid, 1)
    def forward(self, x):
        x=x.unsqueeze(1)
        h=self.tcn(x)
        h=h.mean(-1)
        out=self.head(h).squeeze(-1)
        return out

TCN=None
TCN_CKPT=os.getenv("MODEL_CKPT_TCN","D:/ygy_work/coin/multimodel/tcn_best.pt")
def _load_state_safe(path):
    # 1차: weights_only=True (안전, add_safe_globals 사용)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        LOGGER.warning(f"[TCN][LOAD][WARN] weights_only=True 실패 -> {e}")
        # 2차: 신뢰하는 파일이면 False로 재시도
        return torch.load(path, map_location="cpu", weights_only=False)

try:
    state=_load_state_safe(TCN_CKPT)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state

    class SimpleTCN(torch.nn.Module):
        def __init__(self, in_ch=1, hid=64, ks=3, levels=4):
            super().__init__()
            layers=[]; ch=in_ch
            for i in range(levels):
                layers += [torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(ch, hid, ks, padding=ks//2)
                ), torch.nn.ReLU()]
                ch=hid
            self.tcn=torch.nn.Sequential(*layers)
            self.head=torch.nn.Linear(hid, 1)
        def forward(self, x):
            x=x.unsqueeze(1); h=self.tcn(x); h=h.mean(-1)
            return self.head(h).squeeze(-1)

    TCN=SimpleTCN()
    miss=TCN.load_state_dict(sd, strict=False)
    LOGGER.info(f"[TCN][LOAD] missing={getattr(miss,'missing_keys',[])} "
                f"unexpected={getattr(miss,'unexpected_keys',[])}")
    TCN.eval()
except Exception as e:
    LOGGER.warning(f"[TCN][LOAD][WARN] {e}")
class OnlineAdapter:
    def __init__(self, lr=0.05, maxbuf=240):
        self.a, self.b = 1.0, 0.0
        self.lr=float(lr); self.buf=collections.deque(maxlen=int(maxbuf))
    def transform(self, mu: float) -> float: return self.a*mu + self.b
    def sgd_update(self, mu: float, y: float):
        pred=self.a*mu+self.b; e=pred-y
        self.a -= self.lr*e*mu; self.b -= self.lr*e
        self.a = 0.999*self.a + 0.001*1.0; self.b = 0.999*self.b
        self.buf.append((float(mu), float(y)))

_ADP_DAR: Dict[str, OnlineAdapter] = collections.defaultdict(lambda: OnlineAdapter(0.05, 240))
_ADP_TCN: Dict[str, OnlineAdapter] = collections.defaultdict(lambda: OnlineAdapter(0.05, 240))

def _fee_threshold():
    rt = 2.0*TAKER_FEE + 2.0*(SLIPPAGE_BPS_TAKER/1e4)
    return math.log(1.0 + FEE_SAFETY*rt)

@torch.no_grad()
def deepar_mu(symbol:str)->Optional[float]:
    if DEEPar is None: return None
    df = _recent_1m_df(symbol, minutes=SEQ_LEN + PRED_LEN + 10)
    if df is None or len(df) < (SEQ_LEN + 1): return None
    tsd=_build_infer_dataset(df)
    dl = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
    mu=float(DEEPar.predict(dl, mode="prediction")[0,0])
    mu_adj=_ADP_DAR[symbol].transform(mu) if MODEL_MODE=="online" else mu
    LOGGER.info(f"[PRED][DEEPar] {symbol} mu={mu} mu'={mu_adj} thr={_fee_threshold()}")
    return mu_adj

@torch.no_grad()
def tcn_mu(symbol:str)->Optional[float]:
    if TCN is None: return None
    df=_recent_1m_df(symbol, minutes=SEQ_LEN+5)
    if df is None or len(df)<(SEQ_LEN+1): return None
    x=torch.tensor(df["log_return"].to_numpy()[-SEQ_LEN:], dtype=torch.float32, device=torch.device("cpu")).unsqueeze(0)
    mu=float(TCN(x).cpu().item())
    mu_adj=_ADP_TCN[symbol].transform(mu) if MODEL_MODE=="online" else mu
    return mu_adj

_LAST_SIG = {}
_MIN_SIG_GAP = float(os.getenv("MIN_SIG_GAP", "3.0"))

def _tcn_thr():
    bps = TCN_THR_BPS
    if bps is not None: return float(bps)/1e4
    try: return _fee_threshold()
    except Exception: return 5/1e4

def choose_entry(symbol: str):
    now = time.time()
    if now - _LAST_SIG.get(symbol, 0.0) < _MIN_SIG_GAP:
        return (None, 0.0)
    _LAST_SIG[symbol] = now

    mu_d = deepar_mu(symbol)
    mu_t = tcn_mu(symbol)
    if mu_t is None:
        LOGGER.info(f"[SIG][MISS] {symbol} mu_t=None (TCN not loaded or df short). mu_d={mu_d}")
        return (None, 0.0)
    thr = _tcn_thr()
    side_t = "Buy" if mu_t > 0 else ("Sell" if mu_t < 0 else None)
    side_d = None if mu_d is None else ("Buy" if mu_d > 0 else ("Sell" if mu_d < 0 else None))

    agree = (side_d is not None) and (side_t is not None) and (side_d == side_t)
    strong_t = abs(mu_t) >= thr
    strong_d = (mu_d is not None) and (abs(mu_d) >= thr)

    conf = min(1.0, abs(mu_t)/max(thr,1e-12))
    if agree: conf = min(1.0, conf + 0.25)
    if strong_d: conf = min(1.0, conf + 0.10)

    side = side_t
    mode = (AGREE_MODE or "strict").lower()
    if mode == "strict":
        take = (agree and strong_t)
    elif mode == "soft":
        take = (agree and strong_t) or (strong_t and (side_d is None or not agree))
    elif mode == "tcn_only":
        take = strong_t
    else:
        take = (agree and strong_t)

    if not take:
        LOGGER.info(f"[SIG][SKIP] {symbol} mu_d={mu_d} mu_t={mu_t:.6g} thr={thr:.6g} agree={agree} conf={conf:.2f}")
        return (None, 0.0)

    m = (ENTRY_MODE or "model").lower()
    if m == "inverse":
        side = "Sell" if side == "Buy" else "Buy"
    elif m == "random":
        side = side or random.choice(["Buy","Sell"])

    LOGGER.info(f"[SIG][OK] {symbol} mu_d={mu_d} mu_t={mu_t:.6g} thr={thr:.6g} agree={agree} conf={conf:.2f} side={side}")
    return side, float(conf)

# ===== STATE =====
STATE: Dict[str,dict] = {}
COOLDOWN_UNTIL: Dict[str,float] = {}
PENDING_MAKER: Dict[str,dict] = {}
PLACE_LOCK = threading.Lock()

def _init_state(symbol:str, side:str, entry:float, qty:float, lot_step:float):
    STATE[symbol] = {
        "side":side, "entry":float(entry), "init_qty":float(qty),
        "tp_done":[False,False,False], "be_moved":False,
        "peak":float(entry), "trough":float(entry),
        "entry_ts":time.time(), "lot_step":float(lot_step),
        "deadline_ts": time.time() + _calc_timeout_minutes(symbol)*60
    }

def try_enter(symbol: str, side_override: Optional[str] = None, size_pct: Optional[float] = None):
    try:
        now = time.time()
        if COOLDOWN_UNTIL.get(symbol, 0.0) > now:
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=cooldown until={COOLDOWN_UNTIL[symbol]:.0f}")
            return
        if symbol in PENDING_MAKER:
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=pending_postonly")
            return
        pos_map = get_positions()
        if symbol in pos_map:
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=already_in_position")
            return
        if len(pos_map) >= MAX_OPEN:
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=max_open pos={len(pos_map)}")
            return

        side, conf = choose_entry(symbol)
        if side_override in ("Buy", "Sell"):
            side = side_override
            conf = max(conf or 0.0, 0.60)
        if side not in ("Buy","Sell"):
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=no_signal side={side} conf={conf}")
            return

        bid, ask, mid = get_quote(symbol)
        if not (bid > 0 and ask > 0 and mid > 0):
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=no_quote bid={bid} ask={ask} mid={mid}")
            return

        rule = get_instrument_rule(symbol)
        tick = rule["tickSize"]; lot = rule["qtyStep"]; minq = rule["minOrderQty"]

        entry_ref = float(ask if side == "Buy" else bid)
        tp1_px = _round_to_tick(tp_from_bps(entry_ref, TP1_BPS, side), tick)
        edge_bps = _bps_between(entry_ref, tp1_px)
        need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS

        eq = get_wallet_equity()
        use_pct = float(size_pct if size_pct is not None else ENTRY_EQUITY_PCT)
        qty, _ = compute_qty_by_equity_pct(symbol, mid, eq, use_pct)
        qty = _round_down(qty, lot)

        LOGGER.info(
            f"[ENTER][PREP] {symbol} side={side} mid={mid:.6f} qty={qty} "
            f"edge={edge_bps:.1f} need>={need_bps:.1f} tick={tick} lot={lot}"
        )

        if qty < minq:
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=qty_too_small qty={qty} minq={minq}")
            return
        if edge_bps < need_bps:
            LOGGER.info(f"[ENTER][SKIP] {symbol} reason=edge<need edge={edge_bps:.1f} need={need_bps:.1f}")
            return

        with PLACE_LOCK:
            pos_map = get_positions()
            if len(pos_map) >= MAX_OPEN:
                LOGGER.info(f"[ENTER][LOCK][SKIP] {symbol} reason=max_open pos={len(pos_map)}")
                return
            if symbol in pos_map:
                LOGGER.info(f"[ENTER][LOCK][SKIP] {symbol} reason=already_in_position")
                return
            if COOLDOWN_UNTIL.get(symbol, 0.0) > time.time():
                LOGGER.info(f"[ENTER][LOCK][SKIP] {symbol} reason=cooldown until={COOLDOWN_UNTIL[symbol]:.0f}")
                return

            ensure_isolated_and_leverage(symbol)
            LOGGER.info(f"[ENTER][LOCK] {symbol} exec_mode={EXECUTION_MODE} exit_exec_mode={EXIT_EXEC_MODE}")

            if EXECUTION_MODE == "maker":
                bid, ask, _ = get_quote(symbol)
                if side == "Buy":
                    raw_px = min(bid, max(0.0, ask - 2.0 * tick))
                    safe_px = _round_to_tick(raw_px, tick)
                    if safe_px >= ask:
                        safe_px = _round_to_tick(ask - 2.0 * tick, tick)
                else:
                    raw_px = max(ask, bid + 2.0 * tick)
                    safe_px = _round_to_tick(raw_px, tick)
                    if safe_px <= bid:
                        safe_px = _round_to_tick(bid + 2.0 * tick, tick)

                LOGGER.info(f"[ENTER][POSTONLY][TRY] {symbol} side={side} qty={qty} px={safe_px} bid={bid} ask={ask}")
                oid = place_limit_postonly(symbol, side, qty, safe_px)
                if oid:
                    PENDING_MAKER[symbol] = {"orderId": oid, "ts": time.time(), "side": side, "qty": qty, "price": safe_px}
                    _log_trade({
                        "ts": int(time.time()), "event": "ENTRY_POSTONLY", "symbol": symbol, "side": side,
                        "qty": f"{qty:.8f}", "entry": f"{safe_px:.6f}", "exit": "",
                        "tp1": f"{tp1_px:.6f}", "tp2": "", "tp3": "", "sl": "",
                        "reason": "open", "mode": ENTRY_MODE
                    })
                    LOGGER.info(f"[ENTER][POSTONLY][OK] {symbol} oid={oid}")
                    return
                else:
                    LOGGER.warning(f"[ENTER][POSTONLY][FAIL] {symbol} side={side} qty={qty} px={safe_px} -> fallback TAKER")

            # TAKER
            ok = place_market(symbol, side, qty)
            if not ok:
                LOGGER.warning(f"[ENTER][TAKER][FAIL] {symbol} side={side} qty={qty}")
                return

            pos = get_positions().get(symbol)
            if not pos:
                LOGGER.warning(f"[ENTER][TAKER][NO-POS] {symbol} side={side} qty={qty}")
                return

            entry = float(pos.get("entry") or mid)
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"

            tp1 = _round_to_tick(tp_from_bps(entry, TP1_BPS, side), tick)
            tp2 = _round_to_tick(tp_from_bps(entry, TP2_BPS, side), tick)
            tp3 = _round_to_tick(tp_from_bps(entry, TP3_BPS, side), tick)
            sl  = _round_to_tick(compute_sl(entry, side), tick)

            qty_tp1 = _round_down(qty * TP1_RATIO, lot)
            qty_tp2 = _round_down(qty * TP2_RATIO, lot)

            if qty_tp1 >= minq:
                place_reduce_limit(symbol, close_side, qty_tp1, tp1)
                LOGGER.info(f"[TP1][PLACE]  qty={qty_tp1} px={tp1}")

            if qty_tp2 >= minq and (qty - qty_tp1 - qty_tp2) >= max(0.0, minq - 1e-12):
                place_reduce_limit(symbol, close_side, qty_tp2, tp2)
                LOGGER.info(f"[TP2][PLACE]  qty={qty_tp2} px={tp2}")

            set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=int(pos.get("positionIdx") or 0))


            _init_state(symbol, side, entry, qty, lot)
            _log_trade({
                "ts": int(time.time()), "event": "ENTRY", "symbol": symbol, "side": side,
                "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": "",
                "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                "sl": f"{sl:.6f}", "reason": "open_or_fallback", "mode": ENTRY_MODE
            })
            LOGGER.info(f"[ENTER][TAKER][OK] {symbol} side={side} entry={entry:.6f} qty={qty} "
                        f"tp1={tp1:.6f} tp2={tp2:.6f} tp3={tp3:.6f} sl={sl:.6f}")

    except Exception as e:
        LOGGER.exception(f"[ENTER][ERR] {symbol} -> {e}")

def _fetch_order_status(symbol: str, order_id: str):
    # 최근 주문 조회(히스토리 포함)로 상태 확인
    try:
        r = bybit.get_order_history(category=CATEGORY, symbol=symbol, orderId=order_id)
        lst = ((r.get("result") or {}).get("list") or [])
        if lst: return lst[0]  # {orderStatus, avgPrice, cumExecQty, ...}
    except Exception: pass
    try:
        r = bybit.get_open_orders(category=CATEGORY, symbol=symbol, orderId=order_id)
        lst = ((r.get("result") or {}).get("list") or [])
        if lst: return lst[0]
    except Exception: pass
    return None

def process_pending_postonly():
    now = time.time()
    if not PENDING_MAKER: return
    for sym in list(PENDING_MAKER.keys()):
        oinfo = PENDING_MAKER.get(sym) or {}
        oid   = oinfo.get("orderId"); side = oinfo.get("side")
        st = _fetch_order_status(sym, oid)
        if st is None:
            # 아직 API 집계 안됐을 수 있음 → 조금 더 기다림
            if now - float(oinfo.get("ts", now)) > WAIT_FOR_MAKER_FILL_SEC:
                _cancel_order(sym, oid)
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts": int(time.time()), "event": "CANCEL_POSTONLY", "symbol": sym, "side": side,
                            "qty": "", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
                            "reason": "TIMEOUT_STATUS", "mode": ENTRY_MODE})
                PENDING_MAKER.pop(sym, None)
            continue

        status = st.get("orderStatus")  # 'Filled' / 'Cancelled' / 'New' ...
        if status == "Filled":
            LOGGER.info(f"[POSTONLY][FILLED] {sym} oid={oid}")
            PENDING_MAKER.pop(sym, None)
            # 포지션 확인 → TP/SL 세팅
            pos = get_positions().get(sym)
            if pos:
                entry = float(pos.get("entry") or 0.0)
                rule  = get_instrument_rule(sym)
                tick  = rule["tickSize"]; lot=rule["qtyStep"]; minq=rule["minOrderQty"]
                sideP = pos["side"]; qty=float(pos.get("qty") or 0.0)
                close_side = "Sell" if sideP=="Buy" else "Buy"
                tp1=_round_to_tick(tp_from_bps(entry, TP1_BPS, sideP), tick)
                tp2=_round_to_tick(tp_from_bps(entry, TP2_BPS, sideP), tick)
                tp3=_round_to_tick(tp_from_bps(entry, TP3_BPS, sideP), tick)
                sl = _round_to_tick(compute_sl(entry, sideP), tick)
                qty_tp1=_round_down(qty*TP1_RATIO, lot)
                qty_tp2=_round_down(qty*TP2_RATIO, lot)
                if qty_tp1>=minq: place_reduce_limit(sym, close_side, qty_tp1, tp1)
                if qty_tp2>=minq and (qty-qty_tp1-qty_tp2)>=max(0.0,minq-1e-12):
                    place_reduce_limit(sym, close_side, qty_tp2, tp2)
                set_stop_with_retry(sym, sl_price=sl, tp_price=tp3, position_idx=int(pos.get("positionIdx") or 0))

                _init_state(sym, sideP, entry, qty, lot)
                _log_trade({"ts": int(time.time()), "event": "ENTRY",
                            "symbol": sym, "side": sideP, "qty": f"{qty:.8f}",
                            "entry": f"{entry:.6f}", "exit": "",
                            "tp1": f"{tp1:.6f}","tp2": f"{tp2:.6f}","tp3": f"{tp3:.6f}",
                            "sl": f"{sl:.6f}","reason": "postonly_fill_detect","mode": ENTRY_MODE})
            else:
                # 드물게 체결 직후 포지션 조회 지연 → 쿨다운만
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC

        elif status in ("Cancelled","Rejected"):
            LOGGER.info(f"[POSTONLY][CANCELLED] {sym} oid={oid} status={status}")
            PENDING_MAKER.pop(sym, None)
            COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC

        else:
            # 여전히 열려있음(New)인데 대기 초과면 취소
            if CANCEL_UNFILLED_MAKER and (now - float(oinfo.get("ts", now)) > WAIT_FOR_MAKER_FILL_SEC):
                ok=_cancel_order(sym, oid)
                PENDING_MAKER.pop(sym, None)
                if ok:
                    COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                    _log_trade({"ts": int(time.time()), "event": "CANCEL_POSTONLY", "symbol": sym, "side": side,
                                "qty": "", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
                                "reason": "TIMEOUT", "mode": ENTRY_MODE})
                LOGGER.info(f"[POSTONLY][CANCEL] {sym} oid={oid} after {WAIT_FOR_MAKER_FILL_SEC}s")



def _update_trailing_and_be(symbol:str):
    st=STATE.get(symbol); pos=get_positions().get(symbol)
    if not st or not pos: return
    side=st["side"]; entry=st["entry"]; qty=float(pos.get("qty") or 0.0)
    if qty<=0: return
    _,_,mark = get_quote(symbol)
    st["peak"]=max(st["peak"], mark); st["trough"]=min(st["trough"], mark)
    init_qty=float(st.get("init_qty",0.0)); rem=qty/max(init_qty,1e-12)

    if not st["tp_done"][0] and rem <= (1.0 - TP1_RATIO + 1e-8):
        st["tp_done"][0]=True; _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":symbol,"side":side,
                                           "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"", "reason":"", "mode":ENTRY_MODE})
    if not st["tp_done"][1] and rem <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1]=True; _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":symbol,"side":side,
                                           "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"", "reason":"", "mode":ENTRY_MODE})

    if st["tp_done"][1] and not st.get("be_moved", False):
        be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
        set_trading_stop(symbol, side=side, entry=entry, sl_price=be_px, tp_price=None,
                         pos_idx=int(get_positions()[symbol].get("positionIdx") or 0), mode="Full")

        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":symbol,"side":side,"qty":"","entry":"",
                    "exit":"","tp1":"","tp2":"","tp3":"","sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})

    if st["tp_done"][TRAIL_AFTER_TIER-1]:
        if side=="Buy": new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
        else:           new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
        set_trading_stop(symbol, side=side, entry=entry, sl_price=new_sl, tp_price=None,
                         pos_idx=int(get_positions()[symbol].get("positionIdx") or 0), mode="Full")


def _check_time_stop(symbol: str):
    st = STATE.get(symbol); pos = get_positions().get(symbol)
    if not st or not pos: return
    ddl = st.get("deadline_ts") or (time.time() + _calc_timeout_minutes(symbol) * 60); st["deadline_ts"]=ddl
    if time.time() < ddl: return

    side = pos["side"]; qty = float(pos["qty"])
    if qty <= 0: return
    # Bybit: 시장 청산은 reduceOnly=True로 Market
    rule = get_instrument_rule(symbol)
    qty_str = _fmt_qty(qty, rule["qtyStep"])
    params = dict(category=CATEGORY, symbol=symbol, side=("Sell" if side=="Buy" else "Buy"),
                  orderType="Market", qty=qty_str, reduceOnly=True)
    LOGGER.info(f"[TIMEOUT][EXIT][TRY] {symbol} side={side} -> {params['side']} qty={qty_str}")
    oid = _order_with_mode_retry(params)
    if oid is None:
        LOGGER.warning(f"[TIMEOUT][EXIT][FAIL] {symbol} params={params}")
        st["deadline_ts"] = time.time() + 30
        return
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
    _log_trade({"ts": int(time.time()), "event": "EXIT", "symbol": symbol, "side": side,
                "qty": f"{qty:.8f}", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
                "reason": "HORIZON_TIMEOUT", "mode": ENTRY_MODE})
    LOGGER.info(f"[TIMEOUT][EXIT][OK] {symbol} side={side} qty={qty_str}")
    STATE.pop(symbol, None)

def _update_adapters(symbol:str):
    if MODEL_MODE != "online": return
    df=_recent_1m_df(symbol, minutes=3)
    if df is None or len(df)<2: return
    y = float(np.log(df["close"].iloc[-1] / max(df["close"].iloc[-2], 1e-12)))
    st=STATE.get(symbol); mu_est=0.0
    if st:
        entry=float(st["entry"]); _,_,mark=get_quote(symbol)
        mu_est = float(np.clip(np.log(max(mark,1e-12)/max(entry,1e-12)), -0.01, 0.01))
    _ADP_DAR[symbol].sgd_update(mu_est, y)
    _ADP_TCN[symbol].sgd_update(mu_est, y)

def monitor_loop(poll_sec: float = 1.0):
    LOGGER.info(f"[MONITOR][START] poll={poll_sec}s")
    last_keys=set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            process_pending_postonly()
            pos = get_positions(); keys=set(pos.keys())

            new_syms = keys - last_keys
            for sym in new_syms:
                p=pos[sym]; side=p["side"]; entry=float(p.get("entry") or 0.0); qty=float(p.get("qty") or 0.0)
                if entry<=0 or qty<=0: continue
                lot=get_instrument_rule(sym)["qtyStep"]
                sl = compute_sl(entry, side)
                tp1= tp_from_bps(entry, TP1_BPS, side)
                tp2= tp_from_bps(entry, TP2_BPS, side)
                tp3= tp_from_bps(entry, TP3_BPS, side)
                close_side="Sell" if side=="Buy" else "Buy"
                if _round_down(qty*TP1_RATIO, lot)>0: place_reduce_limit(sym, close_side, _round_down(qty*TP1_RATIO, lot), tp1)
                if _round_down(qty*TP2_RATIO, lot)>0: place_reduce_limit(sym, close_side, _round_down(qty*TP2_RATIO, lot), tp2)
                set_trading_stop(sym, side=side, entry=entry, sl_price=sl, tp_price=tp3,
                                 pos_idx=int(p.get("positionIdx") or 0), mode="Full")
                _init_state(sym, side, entry, qty, lot)
                _log_trade({"ts": int(time.time()), "event": "ENTRY","symbol": sym, "side": side, "qty": f"{qty:.8f}",
                            "entry": f"{entry:.6f}", "exit": "", "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                            "sl": f"{sl:.6f}", "reason": "postonly_fill_or_detect", "mode": ENTRY_MODE})
                LOGGER.info(f"[MONITOR][NEWPOS] {sym} {side} qty={qty} entry={entry:.6f}")

            closed_syms = last_keys - keys
            for sym in closed_syms:
                STATE.pop(sym, None)
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts": int(time.time()), "event": "EXIT","symbol": sym, "side": "", "qty": "",
                            "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
                            "reason": "CLOSED", "mode": ENTRY_MODE})
                LOGGER.info(f"[MONITOR][CLOSED] {sym}")

            for sym in keys:
                _update_trailing_and_be(sym)
                _check_time_stop(sym)
                _update_adapters(sym)

            last_keys = keys
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            LOGGER.info("[MONITOR] stop by user"); break
        except Exception as e:
            LOGGER.exception(f"[MONITOR][ERR] {e}"); time.sleep(2.0)

# ===== SCANNER =====
from concurrent.futures import ThreadPoolExecutor

def _scan_symbol(s):
    LOGGER.debug(f"[SCAN] {s} start")
    if s in PENDING_MAKER:
        LOGGER.debug(f"[SCAN][SKIP] {s} pending_maker"); return
    if len(get_positions()) >= MAX_OPEN:
        LOGGER.debug(f"[SCAN][SKIP] {s} max_open"); return
    if COOLDOWN_UNTIL.get(s,0.0) > time.time():
        LOGGER.debug(f"[SCAN][SKIP] {s} cooldown"); return
    if s in get_positions():
        LOGGER.debug(f"[SCAN][SKIP] {s} already_in_position"); return
    try_enter(s)

def scanner_loop(iter_delay: float = 5.0):
    LOGGER.info(f"[SCANNER][START] delay={iter_delay}s parallel={SCAN_PARALLEL} workers={SCAN_WORKERS}")
    _hb = 0
    while True:
        try:
            _hb += 1
            if _hb % 20 == 0:
                LOGGER.info(f"[HB] scanning... pos={len(get_positions())}")
            syms = SYMBOLS[:]
            if len(get_positions()) < MAX_OPEN:
                if SCAN_PARALLEL and SCAN_WORKERS > 1:
                    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                        ex.map(_scan_symbol, syms)
                else:
                    for s in syms:
                        _scan_symbol(s)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            LOGGER.info("[SCANNER] stop by user"); break
        except Exception as e:
            LOGGER.exception(f"[SCANNER][ERR] {e}"); time.sleep(2.0)

# ===== Run =====
def main():
    setup_live_logger("live_trade_bybit.log", level=logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    print(f"[START] MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} MODEL_MODE={MODEL_MODE} EXCH=Bybit testnet={TESTNET}")
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2=threading.Thread(target=scanner_loop, args=(5.0,), daemon=True)
    t1.start(); t2.start()
    while True: time.sleep(10)

if __name__ == "__main__":
    main()
