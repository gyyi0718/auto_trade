# -*- coding: utf-8 -*-
"""
Bybit Live Trading (Unified v5, Linear Perp)
- DeepAR ckpt 기반 진입 + 수수료/슬리피지 임계
- PostOnly / Taker 진입 지원
- TP1/TP2 지정가(RO), TP3/SL trading_stop
- BE/트레일, 가변 타임아웃, 쿨다운
- ★ 모델 모드:
  * MODEL_MODE=fixed   -> 고정 모델 inference만
  * MODEL_MODE=online  -> 경량 OnlineAdapter(μ' = a*μ + b)로 실시간 보정 + 주기적 OLS 재추정
"""

import os, time, math, csv, threading, logging, random
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch

from torch.serialization import add_safe_globals
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.deepar import DeepAR

try:
    from pybit.unified_trading import HTTP
except Exception as e:
    raise RuntimeError("pip install pybit") from e
# 설정 시점 한 번만


# ====================== ENV ======================
CATEGORY = "linear"
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")          # model|inverse|random
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "maker").lower()   # maker|taker
EXIT_EXEC_MODE = os.getenv("EXIT_EXEC_MODE", "taker").lower()   # maker|taker
MODEL_MODE = os.getenv("MODEL_MODE", "online").lower()          # fixed|online

LEVERAGE   = float(os.getenv("LEVERAGE", "1"))
TAKER_FEE  = float(os.getenv("TAKER_FEE", "0.0006"))
MAKER_FEE  = float(os.getenv("MAKER_FEE", "0.0"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER", "1.0"))
SLIPPAGE_BPS_MAKER = float(os.getenv("SLIPPAGE_BPS_MAKER", "0.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY", "1.20"))
MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "20.0"))

ENTRY_EQUITY_PCT   = float(os.getenv("ENTRY_EQUITY_PCT", "0.20"))
MAX_OPEN   = int(os.getenv("MAX_OPEN", "10"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "10"))
WAIT_FOR_MAKER_FILL_SEC = float(os.getenv("WAIT_FOR_MAKER_FILL_SEC", "30"))
CANCEL_UNFILLED_MAKER = str(os.getenv("CANCEL_UNFILLED_MAKER","1")).lower() in ("1","y","yes","true")

# TP/SL
TP1_BPS, TP2_BPS, TP3_BPS = float(os.getenv("TP1_BPS","50")), float(os.getenv("TP2_BPS","100")), float(os.getenv("TP3_BPS","200"))
TP1_RATIO, TP2_RATIO = float(os.getenv("TP1_RATIO","0.40")), float(os.getenv("TP2_RATIO","0.35"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.01"))
HARD_SL_PCT = float(os.getenv("HARD_SL_PCT","0.01"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))

# Timeout(ATR)
TIMEOUT_MODE = os.getenv("TIMEOUT_MODE", "atr")  # atr|fixed
VOL_WIN = int(os.getenv("VOL_WIN", "60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K", "1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "3"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "20"))
MAX_HOLD_SEC = int(os.getenv("MAX_HOLD_SEC","3600"))

# Model
MODEL_CKPT = os.getenv("MODEL_CKPT", "models/multi_deepar_best.ckpt")
SEQ_LEN    = int(os.getenv("SEQ_LEN","240"))
PRED_LEN   = int(os.getenv("PRED_LEN","60"))

# Online Adapter
ONLINE_LR    = float(os.getenv("ONLINE_LR","0.05"))     # SGD step
ONLINE_DECAY = float(os.getenv("ONLINE_DECAY","0.97"))   # (미사용: 수축은 내부에서 고정)
ONLINE_MAXBUF= int(os.getenv("ONLINE_MAXBUF","60"))     # 버퍼 길이(분)
ADAPTER_REFIT_SEC = int(os.getenv("ADAPTER_REFIT_SEC","120"))  # 주기적 OLS 재적합

# Universe / IO
SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,WLFIUSDT,LINEAUSDT,AVMTUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")
#SYMBOLS= os.getenv("SYMBOLS","SOMIUSDT,WLFIUSDT,AIAUSDT").split(",")

TRADES_CSV = os.getenv("TRADES_CSV","live_trades.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV","live_equity.csv")
TRADES_FIELDS=["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS=["ts","equity"]

SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","6"))

TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
API_KEY    = os.getenv("BYBIT_API_KEY","Dlp4eJD6YFmO99T8vC")
API_SECRET = os.getenv("BYBIT_API_SECRET","YYYB5tMw2TWvfVF5wqi6lQRHqEIiDSpDJF1U")

# ====================== LOGGING ======================
from logging.handlers import RotatingFileHandler
LOGGER = logging.getLogger("live_trade")

LOGGER.propagate = False
for h in list(LOGGER.handlers): LOGGER.removeHandler(h)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
LOGGER.addHandler(ch)
# pybit 내부 로그도 줄이고 싶으면:
logging.getLogger("pybit").setLevel(logging.WARNING)
def setup_live_logger(log_file="live_trade.log", level=logging.INFO):
    LOGGER.setLevel(level); LOGGER.propagate=False
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    for h in list(LOGGER.handlers): LOGGER.removeHandler(h)
    ch=logging.StreamHandler(); ch.setLevel(level); ch.setFormatter(fmt)
    fh=RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level); fh.setFormatter(fmt)
    LOGGER.addHandler(ch); LOGGER.addHandler(fh)
    LOGGER.info("=== Logger initialized ===")

# ====================== HTTP ======================
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET, timeout=10, recv_window=5000)
mkt    = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# ====================== UTIL ======================
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

def get_wallet_equity()->float:
    try:
        wb=client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        rows=(wb.get("result") or {}).get("list") or []
        if not rows: return 0.0
        c = next((x for x in rows[0].get("coin", []) if x.get("coin")=="USDT"), None)
        if not c: return 0.0
        bal=float(c.get("walletBalance") or 0.0); upnl=float(c.get("unrealisedPnl") or 0.0)
        return float(bal+upnl)
    except: return 0.0

def get_positions()->Dict[str,dict]:
    res=client.get_positions(category=CATEGORY, settleCoin="USDT")
    lst=(res.get("result") or {}).get("list") or []
    out={}
    for p in lst:
        sym=p.get("symbol"); size=float(p.get("size") or 0.0)
        if not sym or size<=0: continue
        out[sym]={
            "side":p.get("side"),
            "qty":size,
            "entry":float(p.get("avgPrice") or 0.0),
            "liq":float(p.get("liqPrice") or 0.0),
            "positionIdx":int(p.get("positionIdx") or 0),
        }
    return out

LEV_CAPS: Dict[str, Tuple[float,float]] = {}
def get_symbol_leverage_caps(symbol: str) -> Tuple[float, float]:
    if symbol in LEV_CAPS: return LEV_CAPS[symbol]
    info=client.get_instruments_info(category=CATEGORY, symbol=symbol)
    lst=(info.get("result") or {}).get("list") or []
    if not lst: caps=(1.0, float(LEVERAGE))
    else:
        lf=lst[0].get("leverageFilter") or {}
        caps=(float(lf.get("minLeverage") or 1.0), float(lf.get("maxLeverage") or LEVERAGE))
    LEV_CAPS[symbol]=caps; return caps

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

def _fmt_qty(q: float, step: float) -> str:
    s=str(step); dec=0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{q:.{dec}f}" if dec>0 else str(int(round(q)))).rstrip("0").rstrip(".")

def _fmt_px(p: float, tick: float) -> str:
    s=str(tick); dec=0 if "." not in s else len(s.split(".")[-1].rstrip("0"))
    return (f"{p:.{dec}f}" if dec>0 else str(int(round(p)))).rstrip("0").rstrip(".")

def get_quote(symbol:str)->Tuple[float,float,float]:
    r=mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row=((r.get("result") or {}).get("list") or [{}])[0]
    f=lambda k: float(row.get(k) or 0.0)
    bid,ask,last=f("bid1Price"),f("ask1Price"),f("lastPrice")
    mid=(bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid,ask,mid or 0.0

def ensure_isolated_and_leverage(symbol: str):
    try:
        minL,maxL=get_symbol_leverage_caps(symbol)
        useL=max(min(float(LEVERAGE),maxL),minL)
        kw=dict(category=CATEGORY, symbol=symbol, buyLeverage=str(useL), sellLeverage=str(useL), tradeMode=1)
        try:
            client.set_leverage(**kw)
        except Exception as e:
            msg=str(e)
            if "110043" in msg or "not modified" in msg:
                pass
            else:
                raise
    except Exception as e:
        LOGGER.warning(f"[WARN] set_leverage {symbol}: {e}")

def _order_with_mode_retry(params: dict, side: Optional[str] = None):
    try:
        return client.place_order(**params)
    except Exception as e:
        msg = str(e)
        # 포지션 인덱스 불일치 자동 보정 1회
        if "position idx not match position mode" in msg and side in ("Buy", "Sell"):
            try:
                p2 = dict(params); p2["positionIdx"] = 1 if side == "Buy" else 2
                return client.place_order(**p2)
            except Exception as e2:
                LOGGER.warning(f"[ORDER][REJECT] {params.get('symbol')} {side} -> {str(e2)}", exc_info=False)
                return None
        # 그 외는 스택트레이스 없이 한 줄로만
        LOGGER.warning(f"[ORDER][REJECT] {params.get('symbol')} {side} -> {msg}", exc_info=False)
        return None

# ====================== FEES / FILTERS ======================
def _fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)
_THR = _fee_threshold()

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

# ====================== ORDERS ======================
# === 드랍인: 마켓 주문 (110007 시 축소 재시도, 실패도 한 줄만) ===
def place_market(symbol: str, side: str, qty: float) -> bool:
    try:
        rule = get_instrument_rule(symbol)
        qstep = float(rule["qtyStep"]); minq = float(rule["minOrderQty"])
        q = math.floor(float(qty) / qstep) * qstep
        if q < minq:
            LOGGER.info(f"[ORDER][SKIP] {symbol} {side} qty<{minq} (q={q})"); return False

        for attempt in range(5):
            q = math.floor(q / qstep) * qstep
            if q < minq:
                LOGGER.info(f"[ORDER][ABORT] {symbol} {side} qty dropped below min"); return False
            q_str = _fmt_qty(q, qstep)
            params = dict(category=CATEGORY, symbol=symbol, side=side,
                          orderType="Market", qty=q_str, timeInForce="IOC", reduceOnly=False)
            ret = _order_with_mode_retry(params, side=side)
            if ret:  # 성공
                oid = ((ret or {}).get("result") or {}).get("orderId")
                LOGGER.info(f"[ORDER][MARKET][OK] {symbol} {side} q={q_str} oid={oid}")
                return True

            # 실패 사유에 따라 축소 재시도
            last_msg = "unknown"
            try: last_msg = str(ret)
            except: pass
            if "110007" in last_msg or "ab not enough" in last_msg:
                q *= 0.74; continue
            # 그 외는 즉시 중단(이미 _order_with_mode_retry가 한 줄로 로그 출력)
            return False

        LOGGER.info(f"[ORDER][FAIL] {symbol} {side} market all retries exhausted")
        return False
    except Exception as e:
        LOGGER.warning(f"[ORDER][MARKET][ERR] {symbol} {side} -> {str(e)}", exc_info=False)
        return False

# === 드랍인: PostOnly 지정가 (빈 price 방지 + 110007 축소 + 한 줄만) ===
def place_limit_postonly(symbol: str, side: str, qty: float, price: float) -> Optional[str]:
    try:
        rule = get_instrument_rule(symbol)
        tick = float(rule["tickSize"]); qstep = float(rule["qtyStep"]); minq = float(rule["minOrderQty"])

        # 안전가격(반드시 상대 호가를 교차하지 않게)
        bid, ask, _ = get_quote(symbol)
        if side == "Buy":
            raw_px = min(bid, max(0.0, ask - 2.0*tick))
        else:
            raw_px = max(ask, bid + 2.0*tick)
        px = _round_to_tick(raw_px, tick)
        if side == "Buy" and px >= ask:  px = _round_to_tick(ask - 2.0*tick, tick)
        if side == "Sell" and px <= bid: px = _round_to_tick(bid + 2.0*tick, tick)
        if not (isinstance(px, (int, float)) and px > 0):
            LOGGER.info(f"[ORDER][POSTONLY][ABORT] {symbol} bad px (bid={bid}, ask={ask}, px={px})")
            return None

        q = math.floor(float(qty) / qstep) * qstep
        if q < minq:
            LOGGER.info(f"[ORDER][POSTONLY][SKIP] {symbol} {side} qty<{minq} (q={q})"); return None

        for attempt in range(5):
            q = math.floor(q / qstep) * qstep
            if q < minq:
                LOGGER.info(f"[ORDER][POSTONLY][ABORT] {symbol} qty dropped below min"); return None
            q_str  = _fmt_qty(q, qstep)
            px_str = _fmt_px(px, tick)
            if not px_str:  # 빈 price 방지
                LOGGER.info(f"[ORDER][POSTONLY][ABORT] {symbol} empty px_str"); return None

            params = dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit",
                          qty=q_str, price=px_str, timeInForce="PostOnly", reduceOnly=False)
            LOGGER.info(f"[ORDER][POSTONLY][REQ] {symbol} {side} qty={q_str} px={px_str}")

            ret = _order_with_mode_retry(params, side=side)
            if ret:
                oid = ((ret or {}).get("result") or {}).get("orderId")
                LOGGER.info(f"[ORDER][POSTONLY][OK] {symbol} {side} qty={q_str} px={px_str} oid={oid}")
                return oid

            # 실패 사유에 따라 축소 재시도
            last_msg = "unknown"
            try: last_msg = str(ret)
            except: pass
            if "110007" in last_msg or "ab not enough" in last_msg:
                q *= 0.74; continue
            # 그 외 에러는 이미 _order_with_mode_retry에서 한 줄 출력됨
            return None

        LOGGER.info(f"[ORDER][POSTONLY][FAIL] {symbol} {side} all retries exhausted")
        return None
    except Exception as e:
        LOGGER.warning(f"[ORDER][POSTONLY][ERR] {symbol} {side} -> {str(e)}", exc_info=False)
        return None
def place_reduce_limit(symbol: str, side: str, qty: float, price: float) -> bool:
    try:
        rule=get_instrument_rule(symbol)
        q=_round_down(qty, rule["qtyStep"]); px=_round_to_tick(price, rule["tickSize"])
        if q < rule["minOrderQty"]:
            LOGGER.warning(f"[place_reduce_limit] too small qty -> {symbol} {side} q={q} < min={rule['minOrderQty']}")
            return False
        q_str=_fmt_qty(q, rule["qtyStep"]); px_str=_fmt_px(px, rule["tickSize"])
        params=dict(category=CATEGORY, symbol=symbol, side=side, orderType="Limit",
                    qty=q_str, price=px_str, timeInForce="PostOnly", reduceOnly=True)
        LOGGER.info(f"[ORDER][REDUCE][REQ] {symbol} {side} qty={q_str} px={px_str}")
        ret=_order_with_mode_retry(params, side=side)
        oid=((ret or {}).get("result") or {}).get("orderId")
        LOGGER.info(f"[ORDER][REDUCE][ACK] {symbol} {side} qty={q_str} px={px_str} oid={oid}")
        return True
    except Exception as e:
        LOGGER.exception(f"[place_reduce_limit][ERR] {symbol} {side} {qty}@{price} -> {e}")
        return False

def compute_sl(entry:float, qty:float, side:str)->float:
    dyn = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
    hard= entry*(1.0 - HARD_SL_PCT) if side=="Buy" else entry*(1.0 + HARD_SL_PCT)
    return max(dyn, hard) if side=="Buy" else min(dyn, hard)

def tp_from_bps(entry:float, bps:float, side:str)->float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def set_stop_with_retry(symbol: str, sl_price: Optional[float] = None, tp_price: Optional[float] = None, position_idx: Optional[int] = None) -> bool:
    rule=get_instrument_rule(symbol); tick=float(rule.get("tickSize") or 0.0)
    if sl_price is not None: sl_price=_round_to_tick(float(sl_price), tick)
    if tp_price is not None: tp_price=_round_to_tick(float(tp_price), tick)
    params=dict(category=CATEGORY, symbol=symbol, tpslMode="Full", slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice")
    if sl_price is not None: params["stopLoss"]= _fmt_px(sl_price, tick); params["slOrderType"]="Market"
    if tp_price is not None: params["takeProfit"]=_fmt_px(tp_price, tick); params["tpOrderType"]="Market"
    if position_idx in (1,2): params["positionIdx"]=position_idx
    try:
        LOGGER.info(f"[TRADING_STOP][REQ] {symbol} SL={sl_price} TP={tp_price} posIdx={position_idx}")
        ret=client.set_trading_stop(**params); ok=(ret.get("retCode")==0)
        LOGGER.info(f"[TRADING_STOP][ACK] {symbol} ok={ok} resp={ret}")
        return ok
    except Exception as e:
        LOGGER.exception(f"[TRADING_STOP][ERR] {symbol} -> {e}"); return False

# ====================== QTY ======================
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

# ====================== DATA ======================
def _recent_1m_df(symbol: str, minutes: int = 200):
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=min(minutes,1000))
    rows = (r.get("result") or {}).get("list") or []
    if not rows: return None
    rows = rows[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "series_id": symbol,
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    return df

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

# ====================== MODEL (+ OnlineAdapter) ======================
# ckpt 호환(커스텀 손실 잔재 대비)
class SignAwareNormalLoss(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self,*a,**k): return torch.tensor(0.0)
add_safe_globals([GroupNormalizer, SignAwareNormalLoss])

MODEL=None
try:
    MODEL = DeepAR.load_from_checkpoint(
        MODEL_CKPT,
        target="log_return",
        loss=NormalDistributionLoss(),
        map_location="cpu"
    ).eval()
except Exception as e:
    print(f"[WARN] DeepAR load failed: {e}. Fallback to MA signals.")

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

# ---- Online Adapter (μ' = a*μ + b, SGD + 주기적 OLS 재적합) ----
import collections
class OnlineAdapter:
    def __init__(self, lr=0.05, maxbuf=240):
        self.a, self.b = 1.0, 0.0
        self.lr = float(lr)
        self.buf = collections.deque(maxlen=int(maxbuf))  # (mu, y)
        self.last_refit_ts = 0.0

    def transform(self, mu: float) -> float:
        return self.a*mu + self.b

    def sgd_update(self, mu: float, y: float):
        # y ≈ a*mu + b (MSE)
        pred = self.a*mu + self.b
        e = pred - y
        self.a -= self.lr * e * mu
        self.b -= self.lr * e
        # 약간의 수축(폭주 방지)
        self.a = 0.999*self.a + 0.001*1.0
        self.b = 0.999*self.b
        self.buf.append((float(mu), float(y)))

    def maybe_refit_ols(self, now_ts: float, period_sec: int = 120):
        if now_ts - self.last_refit_ts < period_sec: return
        self.last_refit_ts = now_ts
        if len(self.buf) < 20: return
        mu_arr = np.array([m for m,_y in self.buf], dtype=float)
        y_arr  = np.array([_y for _m,_y in self.buf], dtype=float)
        X = np.vstack([mu_arr, np.ones_like(mu_arr)]).T
        try:
            # OLS: beta = (X^T X)^-1 X^T y
            beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
            a_new, b_new = float(beta[0]), float(beta[1])
            # 안전 클리핑(너무 과격한 계수 방지)
            if np.isfinite(a_new) and np.isfinite(b_new):
                self.a = float(np.clip(a_new, 0.2, 5.0))
                self.b = float(np.clip(b_new, -0.02, 0.02))
        except Exception:
            pass

_ADAPTERS: Dict[str, OnlineAdapter] = collections.defaultdict(lambda: OnlineAdapter(ONLINE_LR, ONLINE_MAXBUF))

def _ma_side(symbol:str)->Optional[str]:
    df=_recent_1m_df(symbol, minutes=120)
    if df is None or len(df)<60: return None
    c=df["close"].to_numpy()
    ma20, ma60 = np.mean(c[-20:]), np.mean(c[-60:])
    if ma20>ma60: return "Buy"
    if ma20<ma60: return "Sell"
    return None

@torch.no_grad()
def deepar_direction(symbol: str):
    side=None; conf=0.0
    try:
        if MODEL is not None:
            df=_recent_1m_df(symbol, minutes=SEQ_LEN+PRED_LEN+10)
            if df is not None and len(df) >= (SEQ_LEN+1):
                tsd=_build_infer_dataset(df)
                dl = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
                mu = float(MODEL.predict(dl, mode="prediction")[0, 0])  # 다음 1분 기대 log-return

                mu_adj = mu
                if MODEL_MODE == "online":
                    mu_adj = _ADAPTERS[symbol].transform(mu)

                thr=_THR
                if   mu_adj >  thr: side="Buy"
                elif mu_adj < -thr: side="Sell"

                LOGGER.info(f"[PRED] {symbol} mu={mu:.6g} mu'={mu_adj:.6g} thr={thr:.6g} -> {side}")
    except Exception as e:
        LOGGER.exception(f"[PRED][ERR] {symbol} -> {e}")

    if side is None:
        side=_ma_side(symbol)
        conf = 0.60 if side else 0.0
    else:
        conf = 1.0

    # entry mode 조정
    m=(ENTRY_MODE or "model").lower()
    if m=="inverse":
        if side=="Buy": side="Sell"
        elif side=="Sell": side="Buy"
    elif m=="random":
        side = side or ( "Buy" if (time.time()*1000)%2<1 else "Sell" )

    return side, float(conf or 0.0)

def choose_entry(symbol:str)->Tuple[Optional[str], float]:
    return deepar_direction(symbol)

# ====================== STATE ======================
STATE: Dict[str,dict] = {}
COOLDOWN_UNTIL: Dict[str,float] = {}
PENDING_MAKER: Dict[str,dict] = {}  # symbol -> {orderId, ts, side, qty, price}
PLACE_LOCK = threading.Lock()

def _init_state(symbol:str, side:str, entry:float, qty:float, lot_step:float):
    STATE[symbol] = {
        "side":side, "entry":float(entry), "init_qty":float(qty),
        "tp_done":[False,False,False], "be_moved":False,
        "peak":float(entry), "trough":float(entry),
        "entry_ts":time.time(), "lot_step":float(lot_step),
        "deadline_ts": time.time() + _calc_timeout_minutes(symbol)*60
    }

# ====================== ENTER ======================
def try_enter(symbol: str, side_override: Optional[str] = None, size_pct: Optional[float] = None):
    try:
        now=time.time()
        if COOLDOWN_UNTIL.get(symbol,0.0) > now: return
        if symbol in PENDING_MAKER: return
        if symbol in get_positions(): return
        if len(get_positions()) >= MAX_OPEN: return

        side, conf = choose_entry(symbol)
        if side_override in ("Buy","Sell"):
            side = side_override
            conf = max(conf, 0.60)
        if side not in ("Buy","Sell") or (conf or 0.0) < 0.55:
            LOGGER.debug(f"[SKIP][SIGNAL] {symbol} side={side} conf={(conf or 0.0):.2f}")
            return

        bid, ask, mid = get_quote(symbol)
        if not (bid>0 and ask>0 and mid>0):
            LOGGER.debug(f"[SKIP][QUOTE] {symbol} bid/ask/mid invalid: {bid},{ask},{mid}")
            return

        rule=get_instrument_rule(symbol); tick=rule["tickSize"]; lot=rule["qtyStep"]; minq=rule["minOrderQty"]
        entry_ref = float(ask if side=="Buy" else bid)
        tp1_px = _round_to_tick(tp_from_bps(entry_ref, TP1_BPS, side), tick)
        edge_bps = _bps_between(entry_ref, tp1_px)
        need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS
        if edge_bps < need_bps:
            LOGGER.debug(f"[SKIP][EDGE] {symbol} edge={edge_bps:.1f} need>={need_bps:.1f}")
            return

        eq=get_wallet_equity()
        use_pct=float(size_pct if size_pct is not None else ENTRY_EQUITY_PCT)
        qty,_=compute_qty_by_equity_pct(symbol, mid, eq, use_pct)
        qty=_round_down(qty, lot)
        if qty < minq:
            LOGGER.debug(f"[SKIP][QTY] {symbol} qty<{minq} (qty={qty}) eq={eq:.2f} mid={mid:.6f}")
            return

        LOGGER.info(f"[ENTER][PREP] {symbol} side={side} conf={(conf or 0.0):.2f} mid={mid:.6f} qty={qty} edge={edge_bps:.1f} need>={need_bps:.1f}")

        with PLACE_LOCK:
            pos_map=get_positions()
            if len(pos_map)>=MAX_OPEN: return
            if COOLDOWN_UNTIL.get(symbol,0.0) > time.time(): return
            if symbol in PENDING_MAKER or symbol in pos_map: return

            ensure_isolated_and_leverage(symbol)

            if EXECUTION_MODE=="maker":
                # ask/bid를 절대 교차하지 않는 안전가격 산출
                if side=="Buy":
                    raw_px = min(bid, max(0.0, ask - 2.0*tick))
                    safe_px = _round_to_tick(raw_px, tick)
                    if safe_px >= ask: safe_px = _round_to_tick(ask - 2.0*tick, tick)
                else:
                    raw_px = max(ask, bid + 2.0*tick)
                    safe_px = _round_to_tick(raw_px, tick)
                    if safe_px <= bid: safe_px = _round_to_tick(bid + 2.0*tick, tick)

                if not (safe_px and safe_px>0):
                    LOGGER.warning(f"[ENTER][POSTONLY][ABORT] {symbol} bad px (raw={raw_px}, safe={safe_px})")
                    return

                oid = place_limit_postonly(symbol, side, qty, safe_px)
                if oid:
                    PENDING_MAKER[symbol] = {"orderId": oid, "ts": time.time(), "side": side, "qty": qty, "price": safe_px}
                    _log_trade({"ts": int(time.time()), "event": "ENTRY_POSTONLY", "symbol": symbol, "side": side,
                                "qty": f"{qty:.8f}", "entry": f"{safe_px:.6f}", "exit": "",
                                "tp1": f"{tp1_px:.6f}", "tp2": "", "tp3": "", "sl": "",
                                "reason": "open", "mode": ENTRY_MODE})
                    LOGGER.info(f"[ENTER][POSTONLY] {symbol} side={side} qty={qty} px={safe_px} oid={oid}")
                else:
                    LOGGER.warning(f"[ENTER][POSTONLY][FAIL] {symbol} side={side} qty={qty}")
                return

            # taker
            ok=place_market(symbol, side, qty)
            if not ok:
                LOGGER.warning(f"[ENTER][TAKER][FAIL] {symbol} side={side} qty={qty}")
                return

            pos=get_positions().get(symbol)
            if not pos:
                LOGGER.warning(f"[ENTER][TAKER][NO-POS] {symbol} side={side} qty={qty}")
                return

            entry=float(pos.get("entry") or mid)
            close_side="Sell" if side=="Buy" else "Buy"

            tp1=_round_to_tick(tp_from_bps(entry, TP1_BPS, side), tick)
            tp2=_round_to_tick(tp_from_bps(entry, TP2_BPS, side), tick)
            tp3=_round_to_tick(tp_from_bps(entry, TP3_BPS, side), tick)
            sl =_round_to_tick(compute_sl(entry, qty, side), tick)

            qty_tp1=_round_down(qty*TP1_RATIO, lot)
            qty_tp2=_round_down(qty*TP2_RATIO, lot)
            if qty_tp1>=minq: place_reduce_limit(symbol, close_side, qty_tp1, tp1)
            if qty_tp2>=minq and (qty - qty_tp1 - qty_tp2) >= max(0.0, minq - 1e-12):
                place_reduce_limit(symbol, close_side, qty_tp2, tp2)

            set_stop_with_retry(symbol, sl_price=sl, tp_price=tp3, position_idx=pos.get("positionIdx"))

            _init_state(symbol, side, entry, qty, lot)
            _log_trade({"ts": int(time.time()), "event": "ENTRY", "symbol": symbol, "side": side,
                        "qty": f"{qty:.8f}", "entry": f"{entry:.6f}", "exit": "",
                        "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                        "sl": f"{sl:.6f}", "reason": "open", "mode": ENTRY_MODE})
            LOGGER.info(f"[ENTER][TAKER][OK] {symbol} side={side} entry={entry:.6f} qty={qty} "
                        f"tp1={tp1:.6f} tp2={tp2:.6f} tp3={tp3:.6f} sl={sl:.6f}")

    except Exception as e:
        LOGGER.exception(f"[ENTER][ERR] {symbol} -> {e}")

# ====================== PENDING POSTONLY ======================
def _get_open_orders(symbol: Optional[str]=None)->List[dict]:
    try:
        res=client.get_open_orders(category=CATEGORY, symbol=symbol)
        return (res.get("result") or {}).get("list") or []
    except Exception:
        return []

def _cancel_order(symbol:str, order_id:str)->bool:
    try:
        client.cancel_order(category=CATEGORY, symbol=symbol, orderId=order_id)
        return True
    except Exception as e:
        LOGGER.warning(f"[CANCEL ERR] {symbol} {order_id}: {e}")
        return False

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

# ====================== MONITOR: BE/Trail + OnlineAdapter 업데이트 ======================
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
        set_stop_with_retry(symbol, sl_price=be_px, tp_price=None, position_idx=pos.get("positionIdx"))
        st["be_moved"]=True
        _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":symbol,"side":side,"qty":"","entry":"",
                    "exit":"","tp1":"","tp2":"","tp3":"","sl":f"{be_px:.6f}","reason":"BE","mode":ENTRY_MODE})

    if st["tp_done"][TRAIL_AFTER_TIER-1]:
        if side=="Buy": new_sl = mark*(1.0 - TRAIL_BPS/10000.0)
        else:           new_sl = mark*(1.0 + TRAIL_BPS/10000.0)
        set_stop_with_retry(symbol, sl_price=new_sl, tp_price=None, position_idx=pos.get("positionIdx"))

def _check_time_stop(symbol:str):
    st=STATE.get(symbol); pos=get_positions().get(symbol)
    if not st or not pos: return
    ddl=st.get("deadline_ts")
    if ddl is None:
        st["deadline_ts"]=time.time()+_calc_timeout_minutes(symbol)*60
        ddl=st["deadline_ts"]
    if time.time() >= ddl:
        side=pos["side"]; qty=float(pos["qty"]); close_side="Sell" if side=="Buy" else "Buy"
        rule=get_instrument_rule(symbol)
        _order_with_mode_retry(dict(
            category=CATEGORY, symbol=symbol, side=close_side, orderType="Market",
            qty=_fmt_qty(qty, rule["qtyStep"]), reduceOnly=True, timeInForce="IOC"
        ), side=close_side)
        COOLDOWN_UNTIL[symbol]=time.time()+COOLDOWN_SEC
        _log_trade({"ts": int(time.time()), "event": "EXIT", "symbol": symbol, "side": side,
                    "qty": f"{qty:.8f}", "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "",
                    "sl": "", "reason": "HORIZON_TIMEOUT", "mode": ENTRY_MODE})
        STATE.pop(symbol, None)

def _update_adapter(symbol:str):
    if MODEL_MODE != "online": return
    # 최근 2틱으로 실제 y(1분 log-return) 산출
    df=_recent_1m_df(symbol, minutes=3)
    if df is None or len(df)<2: return
    y = float(np.log(df["close"].iloc[-1] / max(df["close"].iloc[-2], 1e-12)))
    # μ 추정값: 직전 예측을 저장해두지 않았다면 간략히 현재-엔트리 스케일로 근사(너무 크지 않게 clip)
    st=STATE.get(symbol)
    if st:
        entry=float(st["entry"]); _,_,mark=get_quote(symbol)
        mu_est = float(np.clip(np.log(max(mark,1e-12)/max(entry,1e-12)), -0.01, 0.01))
    else:
        mu_est = 0.0
    _ADAPTERS[symbol].sgd_update(mu_est, y)
    _ADAPTERS[symbol].maybe_refit_ols(time.time(), ADAPTER_REFIT_SEC)

# ====================== LOOPS ======================
from concurrent.futures import ThreadPoolExecutor

def _scan_symbol(s):
    if s in PENDING_MAKER: return
    if len(get_positions()) >= MAX_OPEN: return
    if COOLDOWN_UNTIL.get(s,0.0) > time.time(): return
    if s in get_positions(): return
    try_enter(s)

def scanner_loop(iter_delay: float = 2.5):
    LOGGER.info(f"[SCANNER][START] delay={iter_delay}s parallel={SCAN_PARALLEL} workers={SCAN_WORKERS}")
    while True:
        try:
            syms = SYMBOLS[:]
            if len(get_positions()) < MAX_OPEN:
                if SCAN_PARALLEL:
                    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex: ex.map(_scan_symbol, syms)
                else:
                    for s in syms: _scan_symbol(s)
            time.sleep(iter_delay)
        except KeyboardInterrupt:
            LOGGER.info("[SCANNER] stop by user"); break
        except Exception as e:
            LOGGER.exception(f"[SCANNER][ERR] {e}"); time.sleep(2.0)

def monitor_loop(poll_sec: float = 1.0):
    LOGGER.info(f"[MONITOR][START] poll={poll_sec}s")
    last_keys=set()
    while True:
        try:
            _log_equity(get_wallet_equity())
            process_pending_postonly()
            pos = get_positions(); keys=set(pos.keys())

            # 신규 포지션 초기화(포스트온리 체결 감지 포함)
            new_syms = keys - last_keys
            for sym in new_syms:
                p=pos[sym]; side=p["side"]; entry=float(p.get("entry") or 0.0); qty=float(p.get("qty") or 0.0)
                if entry<=0 or qty<=0: continue
                lot=get_instrument_rule(sym)["qtyStep"]
                sl = compute_sl(entry, qty, side)
                tp1= tp_from_bps(entry, TP1_BPS, side)
                tp2= tp_from_bps(entry, TP2_BPS, side)
                tp3= tp_from_bps(entry, TP3_BPS, side)
                close_side="Sell" if side=="Buy" else "Buy"
                if _round_down(qty*TP1_RATIO, lot)>0: place_reduce_limit(sym, close_side, _round_down(qty*TP1_RATIO, lot), tp1)
                if _round_down(qty*TP2_RATIO, lot)>0: place_reduce_limit(sym, close_side, _round_down(qty*TP2_RATIO, lot), tp2)
                set_stop_with_retry(sym, sl_price=sl, tp_price=tp3, position_idx=p.get("positionIdx"))
                _init_state(sym, side, entry, qty, lot)
                _log_trade({"ts": int(time.time()), "event": "ENTRY","symbol": sym, "side": side, "qty": f"{qty:.8f}",
                            "entry": f"{entry:.6f}", "exit": "", "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
                            "sl": f"{sl:.6f}", "reason": "postonly_fill_or_detect", "mode": ENTRY_MODE})
                LOGGER.info(f"[MONITOR][NEWPOS] {sym} {side} qty={qty} entry={entry:.6f}")

            # 닫힌 포지션
            closed_syms = last_keys - keys
            for sym in closed_syms:
                STATE.pop(sym, None)
                COOLDOWN_UNTIL[sym]=time.time()+COOLDOWN_SEC
                _log_trade({"ts": int(time.time()), "event": "EXIT","symbol": sym, "side": "", "qty": "",
                            "entry": "", "exit": "", "tp1": "", "tp2": "", "tp3": "", "sl": "",
                            "reason": "CLOSED", "mode": ENTRY_MODE})
                LOGGER.info(f"[MONITOR][CLOSED] {sym}")

            # 관리 + 어댑터 업데이트
            for sym in keys:
                _update_trailing_and_be(sym)
                _check_time_stop(sym)
                _update_adapter(sym)

            last_keys = keys
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            LOGGER.info("[MONITOR] stop by user"); break
        except Exception as e:
            LOGGER.exception(f"[MONITOR][ERR] {e}"); time.sleep(2.0)

# ====================== RUN ======================
def main():
    print(f"[START] MODE={ENTRY_MODE} EXEC={EXECUTION_MODE}/{EXIT_EXEC_MODE} MODEL_MODE={MODEL_MODE} TESTNET={TESTNET}")
    t1=threading.Thread(target=monitor_loop, args=(1.0,), daemon=True)
    t2=threading.Thread(target=scanner_loop, args=(2.5,), daemon=True)
    t1.start(); t2.start()
    while True: time.sleep(10)

if __name__ == "__main__":
    setup_live_logger("live_trade.log", level=logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
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
