# -*- coding: utf-8 -*-
"""
Binance Futures Live Trading (USDT-M)
- Dual-Model gating: DeepAR + TCN -> same direction, and |TCN_mu'| >= thr_tcn
- PostOnly(LIMIT GTX) / Market entry
- TP1/TP2 reduce-only limits, TP3/SL with STOP/TP MARKET (closePosition)
- BE move + trailing, ATR timeout, cooldown
"""

import os, time, math, csv, threading, logging, random, collections
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
import torch.nn as nn
# ---------- Logging ----------
from logging.handlers import RotatingFileHandler
from pytorch_forecasting.metrics import NormalDistributionLoss

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
_GLOBAL_BAN_UNTIL = 0.0  # epoch seconds
_KLINE_CACHE = {}        # {symbol: (ts, df)}
_QUOTE_CACHE = {}        # {symbol: (ts, (bid, ask, mid))}
_CACHE_TTL_KLINE = float(os.getenv("CACHE_TTL_KLINE", "5"))   # 초
_CACHE_TTL_QUOTE = float(os.getenv("CACHE_TTL_QUOTE", "1.0")) # 초
_BACKOFF_ON_1003 = float(os.getenv("BACKOFF_ON_1003", "60"))  # 초
# ---------- ENV ----------
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")          # model|inverse|random
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "maker").lower()   # maker|taker
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower()   # maker|taker
MODEL_MODE = os.getenv("MODEL_MODE", "online").lower()          # fixed|online

LEVERAGE   = float(os.getenv("LEVERAGE", "10"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0005"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY", "1.20"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))

MIN_CONF = float(os.getenv("MIN_CONF", "0.30"))            # try_enter에서 요구하는 최소 conf (기존 0.55 → 0.30로 완화)
TCN_THR_BPS = float(os.getenv("TCN_THR_BPS", "3"))         # TCN 임계치(bps). 작을수록 쉽게 진입 (기존 fee기반 → 3bps 기본)
AGREE_MODE = os.getenv("AGREE_MODE", "soft")               # strict | soft | tcn_only
#   strict:  DeepAR & TCN 같은 방향 + TCN 임계치 통과만 진입(기존)
#   soft:    같으면 바로 OK, 다르면 TCN이 강하면 진입
#   tcn_only:DeepAR 없어도/반대여도 TCN만 강하면 진입

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
#SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT").split(",")

TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]
# ENV 기본값 완화 (파일 상단 ENV 섹션 근처)
SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","0")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","1"))
# 스캐너 주기 늘리기
# scanner_loop(iter_delay=2.5) -> 5.0~10.0 권장


# Binance API (python-binance)
from binance.client import Client
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "aFWo0NJ58y7WiXB11R7n2vNv9QYVh1P7YCy0i90GKMfnxPzY9KFTtdISsMutezB6")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "0evwKyLeeJg8djHhCRfbluEnowjCVjMCbGPgB0by0S5tkYi72GU5RhnEBJTkOw6t")
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

import re
from binance.exceptions import BinanceAPIException

def _handle_binance_exc(e: Exception):
    global _GLOBAL_BAN_UNTIL
    if isinstance(e, BinanceAPIException) and e.code == -1003:
        # 메시지에서 'banned until <millis>' 파싱
        m = re.search(r"banned until\s+(\d+)", str(e))
        now = time.time()
        if m:
            until_ms = int(m.group(1))
            _GLOBAL_BAN_UNTIL = max(_GLOBAL_BAN_UNTIL, until_ms / 1000.0)
        else:
            _GLOBAL_BAN_UNTIL = max(_GLOBAL_BAN_UNTIL, now + _BACKOFF_ON_1003)
        wait = max(0.0, _GLOBAL_BAN_UNTIL - now)
        LOGGER.warning(f"[RATE][BAN] -1003 detected. Backing off for ~{wait:.0f}s")
    else:
        LOGGER.debug(f"[RATE][OTHER] {e}")

# 체크포인트가 참조하는 커스텀 클래스 더미 등록
class SignAwareNormalLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, *args, **kwargs):
        # 로드만 통과시키면 되므로 0 텐서 반환
        return torch.tensor(0.0)
# pytorch_forecasting 쪽에서 쓰는 것들도 안전 등록
from pytorch_forecasting.data.encoders import GroupNormalizer
add_safe_globals([GroupNormalizer, SignAwareNormalLoss])

def _tcn_thr():
    # 우선 TCN_THR_BPS 사용, 없으면 수수료기반
    bps = TCN_THR_BPS
    if bps is not None:
        return float(bps)/1e4
    try:
        return _fee_threshold()
    except Exception:
        return 5/1e4  # fallback 5bps
# ---------- CSV IO ----------
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

# ---------- Utils ----------
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

# ---------- Binance helpers ----------
_RULE_CACHE={}
def get_instrument_rule(symbol:str)->dict:
    if symbol in _RULE_CACHE: return _RULE_CACHE[symbol]
    info = client.futures_exchange_info()
    sym = next((x for x in info["symbols"] if x["symbol"]==symbol), None)
    if not sym:
        rule={"tickSize":0.01,"qtyStep":0.001,"minOrderQty":0.001,"minNotional":5.0}
    else:
        pfilter = next(x for x in sym["filters"] if x["filterType"]=="PRICE_FILTER")
        lfilter = next(x for x in sym["filters"] if x["filterType"]=="LOT_SIZE")
        nfilter = next(x for x in sym["filters"] if x["filterType"]=="MIN_NOTIONAL")
        rule = {
            "tickSize": float(pfilter["tickSize"]),
            "qtyStep":  float(lfilter["stepSize"]),
            "minOrderQty": float(lfilter["minQty"]),
            "minNotional": float(nfilter.get("notional", 5.0))
        }
    _RULE_CACHE[symbol]=rule
    return rule

def get_wallet_equity()->float:
    try:
        bal = client.futures_account_balance()
        usdt = next((x for x in bal if x["asset"]=="USDT"), None)
        if not usdt: return 0.0
        return float(usdt.get("balance") or 0.0)
    except Exception:
        return 0.0

def get_positions()->Dict[str,dict]:
    try:
        lst = client.futures_position_information()
    except Exception:
        return {}
    out={}
    for p in lst:
        sym=p["symbol"]; qty=float(p["positionAmt"])
        if abs(qty) <= 0: continue
        side = "Buy" if qty>0 else "Sell"
        out[sym]={
            "side": side,
            "qty": abs(qty),
            "entry": float(p.get("entryPrice") or 0.0),
            "positionIdx": 0,
        }
    return out

def get_symbol_leverage_caps(symbol: str):
    return (1.0, 125.0)

def ensure_isolated_and_leverage(symbol: str):
    try:
        client.futures_change_position_mode(dualSidePosition="false")
    except Exception: pass
    try:
        client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
    except Exception: pass
    try:
        minL,maxL=get_symbol_leverage_caps(symbol)
        useL=max(min(float(LEVERAGE),maxL),minL)
        client.futures_change_leverage(symbol=symbol, leverage=int(useL))
    except Exception: pass

def get_quote(symbol:str)->Tuple[float,float,float]:
    now = time.time()
    # 글로벌 밴이면 빈 값
    if now < _GLOBAL_BAN_UNTIL:
        c = _QUOTE_CACHE.get(symbol)
        if c and (now - c[0] <= _CACHE_TTL_QUOTE):
            return c[1]
        return (0.0, 0.0, 0.0)

    # 캐시 히트
    c = _QUOTE_CACHE.get(symbol)
    if c and (now - c[0] <= _CACHE_TTL_QUOTE):
        return c[1]

    try:
        ob = client.futures_orderbook_ticker(symbol=symbol)
        bid = float(ob.get("bidPrice") or 0.0)
        ask = float(ob.get("askPrice") or 0.0)
        mid = (bid+ask)/2.0 if bid>0 and ask>0 else (bid or ask or 0.0)
        _QUOTE_CACHE[symbol] = (now, (bid, ask, mid))
        return bid, ask, mid
    except Exception as e:
        _handle_binance_exc(e)
        # 마지막 캐시라도 리턴
        c = _QUOTE_CACHE.get(symbol)
        return c[1] if c else (0.0, 0.0, 0.0)

def _recent_1m_df(symbol: str, minutes: int = 200):
    now = time.time()
    limit = min(max(minutes, 1), 1000)

    # 글로벌 밴이면 캐시 사용
    if now < _GLOBAL_BAN_UNTIL:
        c = _KLINE_CACHE.get(symbol)
        return c[1] if c else None

    # 캐시 히트
    c = _KLINE_CACHE.get(symbol)
    if c and (now - c[0] <= _CACHE_TTL_KLINE):
        return c[1]

    try:
        k = client.futures_klines(symbol=symbol, interval="1m", limit=limit)
        if not k:
            return c[1] if c else None
        rows = [{
            "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
            "series_id": symbol,
            "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
            "close": float(z[4]), "volume": float(z[5]),
            "turnover": float(z[7])
        } for z in k]
        df = pd.DataFrame(rows)
        df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
        _KLINE_CACHE[symbol] = (now, df)
        return df
    except Exception as e:
        _handle_binance_exc(e)
        # 마지막 캐시라도 반환
        c = _KLINE_CACHE.get(symbol)
        return c[1] if c else None

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

def compute_qty_by_equity_pct(symbol: str, mid: float, equity_now: float, pct: float) -> Tuple[float, float]:
    rule=get_instrument_rule(symbol)
    qty_step=float(rule["qtyStep"]); min_qty=float(rule["minOrderQty"])
    if mid<=0 or equity_now<=0 or pct<=0: return 0.0, 0.0
    minL,maxL=get_symbol_leverage_caps(symbol)
    lv_used=max(min(float(LEVERAGE),maxL),minL)
    notional=max(equity_now*pct, 10.0)*lv_used
    qty = notional / mid
    qty = math.ceil(qty/qty_step)*qty_step
    if qty < min_qty: qty = math.ceil(min_qty/qty_step)*qty_step
    return qty, qty*mid

# orders
def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        ret = client.futures_create_order(**params)
        return ret
    except Exception as e:
        # 원인 파악을 위해 params/에러 함께 남김
        LOGGER.warning(f"[ORDER][REJECT] {params.get('symbol')} {side} -> {e} | params={params}", exc_info=False)
        return None


def place_market(symbol: str, side: str, qty: float) -> bool:
    rule = get_instrument_rule(symbol)
    q = _round_down(qty, rule["qtyStep"])
    if q < rule["minOrderQty"]:
        return False

    params = dict(
        symbol=symbol,
        side=("BUY" if side == "Buy" else "SELL"),
        type="MARKET",
        quantity=_fmt_qty(q, rule["qtyStep"]),
        # 진입은 reduceOnly 금지 / timeInForce 금지
    )
    ret = _order_with_mode_retry(params, side=side)
    return ret is not None

def place_limit_postonly(symbol: str, side: str, qty: float, price: float):
    rule = get_instrument_rule(symbol)
    q  = _round_down(qty, rule["qtyStep"])
    px = _round_to_tick(price, rule["tickSize"])
    if q < rule["minOrderQty"]:
        LOGGER.info(f"[POSTONLY][SKIP] {symbol} q<{rule['minOrderQty']} q={q}")
        return None

    params = dict(
        symbol=symbol,
        side=("BUY" if side=="Buy" else "SELL"),
        type="LIMIT",
        quantity=_fmt_qty(q, rule["qtyStep"]),
        price=_fmt_px(px, rule["tickSize"]),
        timeInForce="GTX",      # PostOnly
        reduceOnly=False
    )
    LOGGER.info(f"[POSTONLY][TRY] {symbol} side={side} qty={q} px={px}")
    ret = _order_with_mode_retry(params, side=side)
    if ret is None:
        LOGGER.warning(f"[POSTONLY][REJECT] {symbol} side={side} qty={q} px={px}")
        return None
    LOGGER.debug(f"[POSTONLY][RESP] {symbol} -> {ret}")
    return (ret or {}).get("orderId")

def place_reduce_limit(symbol: str, close_side: str, qty: float, price: float) -> bool:
    rule = get_instrument_rule(symbol)
    q  = _round_down(qty, rule["qtyStep"])
    px = _round_to_tick(price, rule["tickSize"])
    if q < rule["minOrderQty"]:
        return False

    pos = get_positions().get(symbol)
    if not pos:
        return False
    pos_qty = float(pos.get("qty") or 0.0)
    if q <= 0 or q - 1e-12 > pos_qty:
        return False

    params = dict(
        symbol=symbol,
        side=_to_side(close_side),      # ← 대문자 정규화
        type="LIMIT",
        quantity=_fmt_qty(q, rule["qtyStep"]),
        price=_fmt_px(px, rule["tickSize"]),
        timeInForce="GTX",
        reduceOnly=True
    )
    ret = _order_with_mode_retry(params, side=params["side"])
    return ret is not None


def _to_side(s: str) -> str:
    return "BUY" if str(s).strip().upper()=="BUY" else "SELL"

def set_stop_with_retry(symbol: str, sl_price: float | None = None, tp_price: float | None = None, close_side: str | None = None) -> bool:
    ok = True
    rule = get_instrument_rule(symbol)

    def _px(p):
        return _fmt_px(_round_to_tick(p, rule["tickSize"]), rule["tickSize"])

    if close_side is None:
        pos = get_positions().get(symbol)
        if not pos:
            return False
        close_side = "SELL" if pos["side"] == "Buy" else "BUY"
    side_u = _to_side(close_side)       # ← 대문자 정규화

    # SL
    if sl_price is not None:
        params = dict(
            symbol=symbol,
            side=side_u,
            type="STOP_MARKET",
            stopPrice=_px(sl_price),
            closePosition=True,
            workingType="MARK_PRICE",
        )
        try:
            client.futures_create_order(**params)   # ← 선물 API
        except Exception as e:
            logging.getLogger("live_trade").warning(f"[SL][ERR] {symbol} -> {e}")
            ok = False

    # TP3
    if tp_price is not None:
        params = dict(
            symbol=symbol,
            side=side_u,
            type="TAKE_PROFIT_MARKET",
            stopPrice=_px(tp_price),
            closePosition=True,
            workingType="MARK_PRICE",
        )
        try:
            client.futures_create_order(**params)   # ← 선물 API
        except Exception as e:
            logging.getLogger("live_trade").warning(f"[TP3][ERR] {symbol} -> {e}")
            ok = False

    return ok

def _get_open_orders(symbol: Optional[str]=None)->List[dict]:
    try:
        if symbol: return client.futures_get_open_orders(symbol=symbol)
        return client.futures_get_open_orders()
    except Exception:
        return []
def _cancel_order(symbol:str, order_id:str)->bool:
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id)
        return True
    except Exception as e:
        LOGGER.warning(f"[CANCEL ERR] {symbol} {order_id}: {e}")
        return False

# ---------- Vol / timeout ----------
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

# ---------- Models (DeepAR + TCN) ----------
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS","1"))))
DEVICE = torch.device("cpu")

# DeepAR
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR

SEQ_LEN = int(os.getenv("SEQ_LEN","120"))
PRED_LEN = int(os.getenv("PRED_LEN","30"))

DEEPar=None
DEEPar_CKPT = os.getenv("MODEL_CKPT_DEEPAR","D:/ygy_work/coin/multimodel/models/multi_deepar_best_main.ckpt")
try:
    DEEPar = DeepAR.load_from_checkpoint(DEEPar_CKPT, map_location="cpu").eval()
    # <-- 핵심: ckpt에 들어있던 loss 무시하고 런타임에서 정상 Loss로 교체
    DEEPar.loss = NormalDistributionLoss()
    DEEPar.to("cpu")
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

# TCN (간단한 1D TCN 분류/회귀 헤드 가정: ckpt에 state_dict['model'] 또는 바로 state_dict)
class SimpleTCN(torch.nn.Module):
    def __init__(self, in_ch=1, hid=64, ks=3, levels=4):
        super().__init__()
        layers=[]
        ch=in_ch
        for i in range(levels):
            layers.append(torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(ch, hid, ks, padding=ks//2)
            ))
            layers.append(torch.nn.ReLU())
            ch=hid
        self.tcn=torch.nn.Sequential(*layers)
        self.head=torch.nn.Linear(hid, 1)
    def forward(self, x):             # x: (B, L)
        x=x.unsqueeze(1)              # (B,1,L)
        h=self.tcn(x)                 # (B,H,L)
        h=h.mean(-1)                  # GAP
        out=self.head(h).squeeze(-1)  # (B,)
        return out                    # μ (다음 log-return 추정치)

TCN=None
TCN_CKPT=os.getenv("MODEL_CKPT_TCN","D:/ygy_work/coin/multimodel/tcn_best.pt")
try:
    state=torch.load(TCN_CKPT, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        sd=state["state_dict"]
    else:
        sd=state
    TCN=SimpleTCN()
    miss=TCN.load_state_dict(sd, strict=False)
    LOGGER.info(f"[TCN][LOAD] missing={miss.missing_keys} unexpected={miss.unexpected_keys}")
    if miss.missing_keys: LOGGER.info(f"[TCN] missing: {miss.missing_keys}")
    if miss.unexpected_keys: LOGGER.info(f"[TCN] unexpected: {miss.unexpected_keys}")
    TCN.eval()
except Exception as e:
    LOGGER.warning(f"[TCN][LOAD][WARN] {e}")

# Online adapters
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
    if df is None:
        LOGGER.info(f"[PRED][DEEPar][MISS] {symbol} -> no df")
        return None
    if len(df) < (SEQ_LEN + 1):
        LOGGER.info(f"[PRED][DEEPar][MISS] {symbol} -> rows={len(df)} need>={SEQ_LEN + 1}")
        return None
    if df is None or len(df)<(SEQ_LEN+1): return None
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
    x=torch.tensor(df["log_return"].to_numpy()[-SEQ_LEN:], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,L)
    mu=float(TCN(x).cpu().item())
    mu_adj=_ADP_TCN[symbol].transform(mu) if MODEL_MODE=="online" else mu
    return mu_adj


_LAST_SIG = {}  # {symbol: last_ts}
_MIN_SIG_GAP = float(os.getenv("MIN_SIG_GAP", "3.0"))


def choose_entry(symbol: str):
    now = time.time()
    if now - _LAST_SIG.get(symbol, 0.0) < _MIN_SIG_GAP:
        return (None, 0.0)
    _LAST_SIG[symbol] = now

    mu_d = deepar_mu(symbol)   # None일 수 있음
    mu_t = tcn_mu(symbol)      # None이면 진입 불가
    if mu_t is None:
        return (None, 0.0)

    thr = _tcn_thr()

    side_t = "Buy" if mu_t > 0 else ("Sell" if mu_t < 0 else None)
    side_d = None if mu_d is None else ("Buy" if mu_d > 0 else ("Sell" if mu_d < 0 else None))

    agree = (side_d is not None) and (side_t is not None) and (side_d == side_t)
    strong_t = abs(mu_t) >= thr
    strong_d = (mu_d is not None) and (abs(mu_d) >= thr)

    # 신뢰도(conf): TCN 세기 + 합의 보너스
    conf = min(1.0, abs(mu_t)/max(thr,1e-12))
    if agree: conf = min(1.0, conf + 0.25)     # 합의 시 가점
    if strong_d: conf = min(1.0, conf + 0.10)  # DeepAR도 강하면 가점

    side = side_t  # 기본은 TCN 방향

    mode = (AGREE_MODE or "strict").lower()
    take = False
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

    # ENTRY_MODE 반영
    m = (ENTRY_MODE or "model").lower()
    if m == "inverse":
        side = "Sell" if side == "Buy" else "Buy"
    elif m == "random":
        side = side or random.choice(["Buy","Sell"])

    LOGGER.info(f"[SIG][OK] {symbol} mu_d={mu_d} mu_t={mu_t:.6g} thr={thr:.6g} agree={agree} conf={conf:.2f} side={side}")
    return side, float(conf)
# ---------- STATE ----------
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
            # LOCK 내 재검사 + 사유 로깅
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
                # 안전 GTX 가격 계산 & 로깅
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

            # TAKER 진입(기본 또는 GTX 실패 폴백)
            ok = place_market(symbol, side, qty)
            if not ok:
                LOGGER.warning(f"[ENTER][TAKER][FAIL] {symbol} side={side} qty={qty}")
                return

            # 포지션 확인 후 TP/SL 세팅
            pos = get_positions().get(symbol)
            if not pos:
                LOGGER.warning(f"[ENTER][TAKER][NO-POS] {symbol} side={side} qty={qty}")
                return

            entry = float(pos.get("entry") or mid)
            close_side = "SELL" if pos["side"] == "Buy" else "BUY"

            tp1 = _round_to_tick(tp_from_bps(entry, TP1_BPS, side), tick)
            tp2 = _round_to_tick(tp_from_bps(entry, TP2_BPS, side), tick)
            tp3 = _round_to_tick(tp_from_bps(entry, TP3_BPS, side), tick)
            sl  = _round_to_tick(compute_sl(entry, side), tick)

            qty_tp1 = _round_down(qty * TP1_RATIO, lot)
            qty_tp2 = _round_down(qty * TP2_RATIO, lot)

            if qty_tp1 >= minq:
                place_reduce_limit(symbol, close_side, qty_tp1, tp1)
            if qty_tp2 >= minq and (qty - qty_tp1 - qty_tp2) >= max(0.0, minq - 1e-12):
                place_reduce_limit(symbol, close_side, qty_tp2, tp2)

            set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, close_side=close_side)

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

# ---------- PENDING POSTONLY ----------
def process_pending_postonly():
    now=time.time()
    if not PENDING_MAKER: return
    open_map: Dict[str, List[dict]] = {}
    for sym in list(PENDING_MAKER.keys()):
        if sym not in open_map:
            open_map[sym] = _get_open_orders(sym)
        oinfo=PENDING_MAKER.get(sym) or {}
        oid=oinfo.get("orderId"); side=oinfo.get("side")
        open_list=open_map.get(sym) or []
        still_open = any((x.get("orderId")==oid) for x in open_list)
        if not still_open:
            LOGGER.info(f"[POSTONLY][FILLED/REMOVED] {sym} oid={oid}")
            PENDING_MAKER.pop(sym, None)
            continue
        if CANCEL_UNFILLED_MAKER and (now - float(oinfo.get("ts", now)) > WAIT_FOR_MAKER_FILL_SEC):
            ok=_cancel_order(sym, oid)
            PENDING_MAKER.pop(sym, None)
            if ok:
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts": int(time.time()), "event": "CANCEL_POSTONLY", "symbol": sym, "side": side,
                            "qty": "", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
                            "reason": "TIMEOUT", "mode": ENTRY_MODE})
                LOGGER.info(f"[POSTONLY][CANCEL] {sym} oid={oid} after {WAIT_FOR_MAKER_FILL_SEC}s")

# ---------- MONITOR ----------
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
                                           "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"",
                                           "reason":"","mode":ENTRY_MODE})
    if not st["tp_done"][1] and rem <= (1.0 - TP1_RATIO - TP2_RATIO + 1e-8):
        st["tp_done"][1]=True; _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":symbol,"side":side,
                                           "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"","sl":"",
                                           "reason":"","mode":ENTRY_MODE})

    if st["tp_done"][1] and not st.get("be_moved", False):
        be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=be_px, tp_price=None, close_side=("Sell" if side=="Buy" else "Buy"))
        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":symbol,"side":side,"qty":"","entry":"",
                    "exit":"","tp1":"","tp2":"","tp3":"","sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})

    if st["tp_done"][TRAIL_AFTER_TIER-1]:
        if side=="Buy": new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
        else:           new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=new_sl, tp_price=None, close_side=("Sell" if side=="Buy" else "Buy"))

def _check_time_stop(symbol: str):
    st = STATE.get(symbol); pos = get_positions().get(symbol)
    if not st or not pos:
        return

    ddl = st.get("deadline_ts")
    if ddl is None:
        st["deadline_ts"] = time.time() + _calc_timeout_minutes(symbol) * 60
        ddl = st["deadline_ts"]

    if time.time() >= ddl:
        side = pos["side"]
        qty = float(pos["qty"])
        if qty <= 0:
            return

        close_side = "Sell" if side == "Buy" else "Buy"
        side_param = "SELL" if close_side == "Sell" else "BUY"

        rule = get_instrument_rule(symbol)
        qty_str = _fmt_qty(qty, rule["qtyStep"])

        params = dict(
            symbol=symbol,
            side=side_param,
            type="MARKET",
            quantity=qty_str,
            reduceOnly=True,   # ← 시간 청산은 청산용이므로 유지
            # ❌ timeInForce 넣지 말 것 (MARKET과 함께 쓰면 -1106)
        )

        LOGGER.info(f"[TIMEOUT][EXIT][TRY] {symbol} side={side} -> {side_param} qty={qty_str}")
        ok = _order_with_mode_retry(params, side=close_side) is not None
        if not ok:
            LOGGER.warning(f"[TIMEOUT][EXIT][FAIL] {symbol} params={params}")
            # 실패 시, 너무 자주 반복 방지
            st["deadline_ts"] = time.time() + 30
            return

        COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
        _log_trade({
            "ts": int(time.time()), "event": "EXIT", "symbol": symbol, "side": side,
            "qty": f"{qty:.8f}", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "",
            "sl": "", "reason": "HORIZON_TIMEOUT", "mode": ENTRY_MODE
        })
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
                set_stop_with_retry(sym, sl_price=sl, tp_price=tp3, close_side=close_side)
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

# ---------- SCANNER ----------
from concurrent.futures import ThreadPoolExecutor

def _scan_symbol(s):
    LOGGER.debug(f"[SCAN] {s} start")
    if s in PENDING_MAKER:
        LOGGER.debug(f"[SCAN][SKIP] {s} pending_maker")
        return
    if len(get_positions()) >= MAX_OPEN:
        LOGGER.debug(f"[SCAN][SKIP] {s} max_open")
        return
    if COOLDOWN_UNTIL.get(s,0.0) > time.time():
        LOGGER.debug(f"[SCAN][SKIP] {s} cooldown")
        return
    if s in get_positions():
        LOGGER.debug(f"[SCAN][SKIP] {s} already_in_position")
        return
    try_enter(s)

from concurrent.futures import ThreadPoolExecutor

def scanner_loop(iter_delay: float = 2.5):
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

# ---------- RUN ----------
def main():
    setup_live_logger("live_trade.log", level=logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    print(f"[START] MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} MODEL_MODE={MODEL_MODE} EXCH=Binance")
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2 = threading.Thread(target=scanner_loop, args=(5.0,), daemon=True)  # 2.5 -> 5.0
    t1.start(); t2.start()
    while True: time.sleep(10)

if __name__ == "__main__":
    main()
