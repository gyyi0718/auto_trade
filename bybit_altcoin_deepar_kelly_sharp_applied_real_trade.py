
# -*- coding: utf-8 -*-
"""
bybit_altcoin_deepar_kelly_sharp_applied_real_trade.py (model-direction profit friendly)
- Multi-TP: TP1/TP2/TP3 (default: 0.15% / 0.5% / 1.0%)
- Partial exits: TP1_RATIO / TP2_RATIO (remainder at TP3)
- Delayed BE: move SL to BE(+eps) AFTER TP2 (configurable)
- Wider Trailing: default 40 bps, activates after configurable tier
- Relative TP: optional TAKE_PROFIT_REL_PCT (e.g., 0.5% of notional)
- Fast-start: DeepAR is OFF by default (USE_DEEPAR=0)
"""

import os, time, math, logging, warnings, csv, pathlib, random
from typing import Dict, Tuple, Optional, List
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ===================== 사용자 설정 =====================
def _envf(name: str, default: float) -> float:
    v = os.getenv(name);
    return float(v) if v is not None and v.strip() != "" else float(default)

def _envi(name: str, default: int) -> int:
    v = os.getenv(name);
    return int(v) if v is not None and v.strip() != "" else int(default)

def _envs(name: str, default: str) -> str:
    v = os.getenv(name);
    return v if v is not None and v.strip() != "" else default

API_KEY    = ""
API_SECRET = ""
TESTNET    = _envs("BYBIT_TESTNET", "false").lower() == "true"

# 모드: "model" | "inverse" | "random"
MODE = _envs("MODE", "model")
CSV_SAVE_MODE = _envs("CSV_SAVE_MODE", "single")  # "single" | "split"

# 카테고리/거래 공통
CATEGORY   = "linear"
LEVERAGE   = _envi("LEVERAGE", 20)
TAKER_FEE  = _envf("TAKER_FEE", 0.0006)

# ===== 개선된 TP/SL 파라미터 =====
TP1_BPS        = _envf("TP1_BPS", 15.0)        # +0.15%
TP2_BPS        = _envf("TP2_BPS", 50.0)        # +0.50%
TP3_BPS        = _envf("TP3_BPS", 100.0)       # +1.00%
TP1_RATIO      = _envf("TP1_RATIO", 0.25)      # 25% 청산
TP2_RATIO      = _envf("TP2_RATIO", 0.35)      # 35% 청산
# 나머지 비율은 자동 (TP3에서 전량)

BE_EPS_BPS     = _envf("BE_EPS_BPS", 2.0)      # BE + 2bps
BE_AFTER_TIER  = _envi("BE_AFTER_TIER", 2)     # 1=TP1, 2=TP2, 3=TP3

TRAIL_BPS      = _envf("TRAIL_BPS", 40.0)      # 넉넉하게 추적
TRAIL_AFTER_TIER = _envi("TRAIL_AFTER_TIER", 1)# 어느 티어 후 트레일링 시작? (기본 TP1)

MAX_HOLD_SEC   = _envi("MAX_HOLD_SEC", 3600)   # 추세에 기회 주기 (기본 1시간)
COOLDOWN_SEC   = _envi("COOLDOWN_SEC", 60)

TAKE_PROFIT_USD= _envf("TAKE_PROFIT_USD", 0.0) # 절대 USD TP (0이면 비활성)
TAKE_PROFIT_REL_PCT = _envf("TAKE_PROFIT_REL_PCT", 0.005) # notional * 0.5%

STOP_LOSS_PCT  = _envf("STOP_LOSS_PCT", 0.03)  # 초기 고정 SL 3%

# 품질/체결 가드
SPREAD_REJECT_BPS = _envf("SPREAD_REJECT_BPS", 35.0)
GAP_REJECT_BPS    = _envf("GAP_REJECT_BPS", 400.0)

# 리스크 설정
RISK_PCT_OF_EQUITY     = _envf("RISK_PCT_OF_EQUITY", 0.30)
ENTRY_PORTION          = _envf("ENTRY_PORTION", 0.40)
MAX_OPEN               = _envi("MAX_OPEN", 4)

# 모델/심볼
CKPT_PATH = _envs("CKPT_PATH", "models/multi_deepar_model.ckpt")
USE_DEEPAR = _envs("USE_DEEPAR", "0") == "1"  # 기본 꺼짐
SEQ_LEN   = _envi("SEQ_LEN", 60)
PRED_LEN  = _envi("PRED_LEN", 10)

SYMBOLS = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]
EXCLUDE_SYMBOLS: List[str] = []

ALLOW_TRADING_WINDOWS_LOCAL: Optional[List[Tuple[int,int]]] = None

# ===================== 로깅 파일 =====================
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_FIELDS = [
    "ts","event","symbol","side","qty",
    "entry_price","tp_price","sl_price",
    "order_id","tp_order_id","sl_order_id",
    "reason","pnl_usdt","roi","extra","mode"
]

def _trade_csv_path(symbol: str) -> pathlib.Path:
    if CSV_SAVE_MODE == "split":
        p = LOG_DIR / f"trades_{symbol}.csv"
        if not p.exists():
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(TRADES_FIELDS)
        return p
    else:
        p = LOG_DIR / f"live_trades_leverage{LEVERAGE}.csv"
        if not p.exists():
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(TRADES_FIELDS)
        return p

EQUITY_CSV = LOG_DIR / f"live_equity_leverage{LEVERAGE}.csv"
if not EQUITY_CSV.exists():
    with open(EQUITY_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ts","cash","upnl","equity"])

def _write_trade_row(row: dict):
    path = _trade_csv_path(row.get("symbol",""))
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            row.get("ts",""), row.get("event",""), row.get("symbol",""),
            row.get("side",""), row.get("qty",""),
            row.get("entry_price",""), row.get("tp_price",""), row.get("sl_price",""),
            row.get("order_id",""), row.get("tp_order_id",""), row.get("sl_order_id",""),
            row.get("reason",""), row.get("pnl_usdt",""), row.get("roi",""),
            row.get("extra",""), row.get("mode", MODE),
        ])

_last_eq_log_ts = 0.0
def _log_equity_snapshot(get_equity_breakdown):
    global _last_eq_log_ts
    now = time.time()
    if now - _last_eq_log_ts < 3.0:
        return
    eq, cash, upnl, _ = get_equity_breakdown()
    with open(EQUITY_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([int(now), f"{cash:.6f}", f"{upnl:+.6f}", f"{eq:.6f}"])
    _last_eq_log_ts = now

# ===================== Bybit 클라이언트 =====================
from pybit.unified_trading import HTTP
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, recv_window=300000, timeout=10, testnet=TESTNET)

# ===================== 유틸 =====================
def _d(x)->Decimal: return Decimal(str(x))
def _round_step(x: Decimal, step: Decimal)->Decimal:
    if step <= 0: step = Decimal("0.001")
    return (x / step).to_integral_value(rounding=ROUND_DOWN) * step

def fnum(x, default=0.0)->float:
    try:
        if x is None: return float(default)
        s = str(x).strip().lower()
        if s in {"","nan","none","null","inf","-inf"}: return float(default)
        return float(s)
    except Exception:
        return float(default)

def _bps_from_to(a: float, b: float) -> float:
    if a <= 0: return 0.0
    return (b - a) / a * 10_000.0

def _opp(side: str) -> str:
    return "Sell" if side == "Buy" else "Buy"

def _roi(side: str, entry: float, price: float) -> float:
    return (price/entry - 1.0) if side=="Buy" else (entry/price - 1.0)

def _local_hour_kst(ts: Optional[float]=None) -> int:
    t = ts if ts is not None else time.time()
    return int((time.gmtime(t).tm_hour + 9) % 24)

def trading_allowed_now() -> bool:
    if not ALLOW_TRADING_WINDOWS_LOCAL:
        return True
    h = _local_hour_kst()
    for start, end in ALLOW_TRADING_WINDOWS_LOCAL:
        if start == end:  # 24h
            return True
        if start < end:
            if start <= h < end: return True
        else:  # e.g., 22~02
            if h >= start or h < end: return True
    return False

# ===================== 시세/심볼 정보 =====================
def get_last_price(symbol: str)->float:
    res = client.get_tickers(category=CATEGORY, symbol=symbol)
    row = (res.get("result", {}) or {}).get("list", [{}])[0]
    for k in ("lastPrice","markPrice","bid1Price"):
        p = fnum(row.get(k), 0.0)
        if p > 0: return p
    raise RuntimeError(f"{symbol} 가격 조회 실패: {row}")

def get_quote(symbol: str) -> Tuple[float, float, float]:
    res = client.get_tickers(category=CATEGORY, symbol=symbol)
    row = (res.get("result", {}) or {}).get("list", [{}])[0]
    bid = fnum(row.get("bid1Price"), 0.0)
    ask = fnum(row.get("ask1Price"), 0.0)
    last = fnum(row.get("lastPrice"), 0.0)
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        return bid, ask, mid
    p = last if last > 0 else get_last_price(symbol)
    return p, p, p

def fetch_recent_data(symbol: str, limit: int = 120)->pd.DataFrame:
    res = client.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=limit)
    rows = (res.get("result", {}) or {}).get("list", [])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df["series_id"] = symbol
    return df

def compute_sigma_pct_from_df(df: pd.DataFrame, lookback=120)->float:
    s = df.tail(lookback)["close"]
    lr = np.log(s).diff().dropna().values
    if lr.size == 0: return 0.003
    return float(np.std(lr))

# ===================== DeepAR (옵션) =====================
_deepar_model = None
def _load_deepar():
    global _deepar_model
    if not USE_DEEPAR:
        return False
    if _deepar_model is None:
        try:
            from pytorch_forecasting import DeepAR
            _deepar_model = DeepAR.load_from_checkpoint(CKPT_PATH)
        except Exception as e:
            print("[WARN] DeepAR load failed:", e)
            _deepar_model = False
    return _deepar_model

def _extract_quantile_array(pred_out):
    import numpy as _np, torch as _torch
    if isinstance(pred_out, dict):
        for k in ("prediction","predictions","output"):
            if k in pred_out: pred_out = pred_out[k]; break
    if isinstance(pred_out, list):
        pred_out = _np.array(pred_out)
    if isinstance(pred_out, _torch.Tensor):
        arr = pred_out.detach().cpu().numpy()
    elif isinstance(pred_out, _np.ndarray):
        arr = pred_out
    else:
        arr = _np.array(pred_out)
    if arr.ndim == 2:
        if arr.shape[1] in (3,5): arr = arr[None,:,:]
        elif arr.shape[0] in (3,5): arr = arr.transpose(1,0)[None,:,:]
        else: arr = arr[None,:,:]
    elif arr.ndim == 3:
        B,A,C = arr.shape
        if C in (3,5): pass
        elif A in (3,5): arr = arr.transpose(0,2,1)
    else:
        raise RuntimeError(f"Unexpected prediction shape: {arr.shape}")
    return arr

def adjust_tp_preview(side: str, now_price: float, predicted_target: float)->float:
    step = 1.0 + (TP1_BPS / 10_000.0)
    if side == "Buy":
        min_tp = round(now_price * step, 6)
        return max(predicted_target, min_tp)
    else:
        max_tp = round(now_price / step, 6)
        return min(predicted_target, max_tp)

def _predict_with_deepar(symbol: str, df: pd.DataFrame)->Tuple[float,int,float]:
    now_price = float(df["close"].iloc[-1])
    model = _load_deepar()
    if model is False:
        return now_price, 60, 0.0

    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    x = df.sort_values("timestamp").copy()
    x["time_idx"] = np.arange(len(x))
    x["series_id"] = symbol
    x["log_return"] = np.log(x["close"]).diff().fillna(0.0)

    enc = x.iloc[-SEQ_LEN:].copy()
    enc["time_idx"] -= enc["time_idx"].min()

    fut = pd.DataFrame({"time_idx": np.arange(SEQ_LEN, SEQ_LEN+PRED_LEN),
                        "series_id": symbol, "log_return": [0.0]*PRED_LEN})
    comb = pd.concat([enc, fut], ignore_index=True)

    ds = TimeSeriesDataSet(
        comb, time_idx="time_idx", target="log_return", group_ids=["series_id"],
        max_encoder_length=SEQ_LEN, max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    dl = ds.to_dataloader(train=False, batch_size=1)

    try:
        q_raw = model.predict(dl, mode="quantiles", mode_kwargs={"quantiles":[0.1,0.5,0.9]}, return_x=False)
    except TypeError:
        q_raw = model.predict(dl, mode="quantiles", quantiles=[0.1,0.5,0.9], return_x=False)

    try:
        q_arr = _extract_quantile_array(q_raw)
        q10 = q_arr[0,:,0]; q50 = q_arr[0,:,1]; q90 = q_arr[0,:,2]
    except Exception:
        pred = model.predict(dl, mode="prediction", return_x=False)
        q_arr = _extract_quantile_array(pred)
        q10 = q_arr[0,:,0]; q50 = q_arr[0,:,0]; q90 = q_arr[0,:,0]

    def to_price_path(logret_vec):
        lr_cum = np.cumsum(logret_vec)
        return now_price * np.exp(lr_cum)

    p10 = to_price_path(q10); p50 = to_price_path(q50); p90 = to_price_path(q90)

    best_idx, best_gain, best_tp = 0, -1.0, None
    for i in range(len(p50)):
        raw_tp = float(p50[i])
        side = "Buy" if raw_tp > now_price else "Sell"
        tp_adj = adjust_tp_preview(side, now_price, raw_tp)
        gain = abs(tp_adj - now_price)
        if gain > best_gain:
            best_gain, best_idx, best_tp = gain, i, tp_adj

    band = float(p90[best_idx] - p10[best_idx])
    rel_unc = band / max(now_price, 1e-9)
    confidence = float(np.clip(1.0 - (rel_unc / 0.01), 0.0, 1.0))
    target_sec = (best_idx + 1) * 60
    return float(best_tp), int(target_sec), confidence

def _predict_with_momentum(symbol: str, df: pd.DataFrame)->Tuple[float,int,float]:
    now = float(df["close"].iloc[-1])
    s = df["close"].values
    ma_fast = s[-10:].mean() if len(s) >= 10 else s.mean()
    ma_slow = s[-30:].mean() if len(s) >= 30 else s.mean()
    drift = np.sign(s[-1] - s[-5]) if len(s) >= 5 else 0.0
    bias = 1 if (ma_fast > ma_slow and drift >= 0) else -1
    step = 1.0 + (TP1_BPS/10_000.0)
    target = now * (step if bias>0 else 1.0/step)
    conf = 0.80 if bias != 0 else 0.60
    return float(target), 60, conf

def predict_future_price(symbol: str, df: pd.DataFrame)->Tuple[float,int,float]:
    if USE_DEEPAR:
        return _predict_with_deepar(symbol, df)
    else:
        return _predict_with_momentum(symbol, df)

# ===================== Bybit helpers =====================
def ensure_leverage(symbol: str, leverage: int):
    try:
        client.set_leverage(category=CATEGORY, symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))
    except Exception as e:
        print("[WARN] set_leverage:", e)

def get_symbol_filters(symbol: str)->Tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    """qty_step, min_qty, max_qty, min_notional, tick"""
    info = client.get_instruments_info(category=CATEGORY, symbol=symbol)
    row = (info.get("result",{}) or {}).get("list",[{}])[0]
    lotSz = _d(row.get("lotSizeFilter",{}).get("qtyStep","0.001"))
    minQty = _d(row.get("lotSizeFilter",{}).get("minOrderQty","0.0"))
    maxQty = _d(row.get("lotSizeFilter",{}).get("maxOrderQty","999999999"))
    minNotional = _d(row.get("lotSizeFilter",{}).get("minNotionalValue","0.0"))
    tickSz = _d(row.get("priceFilter",{}).get("tickSize","0.01"))
    return lotSz, minQty, maxQty, minNotional, tickSz

def _posidx() -> dict:
    return {"positionIdx": 0}

def _max_qty_from_avail(use_usdt: Decimal, price: Decimal, lev: int, qty_step: Decimal, min_qty: Decimal)->Decimal:
    notional = use_usdt * lev
    qty = _round_step(notional / max(price, _d("1e-9")), qty_step)
    if qty < min_qty: qty = min_qty
    return qty

def place_market_safely(symbol: str, side: str, use_usdt: float, leverage: int)->Tuple[Optional[str], Optional[Decimal], Optional[Decimal]]:
    ensure_leverage(symbol, leverage)
    bid, ask, mid = get_quote(symbol)

    # 스프레드 가드
    if bid > 0 and ask > 0:
        spread_bps = (ask - bid) / max(mid, 1e-9) * 10_000.0
        if spread_bps > SPREAD_REJECT_BPS:
            print(f"[SKIP] {symbol} spread={spread_bps:.2f}bps > {SPREAD_REJECT_BPS}bps")
            return None, None, None

    ref_price = _d(ask if side=="Buy" else bid)
    qty_step, min_qty, max_qty, min_notional, tick = get_symbol_filters(symbol)
    use = _d(use_usdt)
    if use <= 0:
        return None, None, None

    qty = _max_qty_from_avail(use, ref_price, leverage, qty_step, min_qty)
    if min_notional > 0 and qty * ref_price < min_notional:
        qty = _round_step((min_notional / ref_price) + qty_step, qty_step)
    qty = min(qty, max_qty)
    if qty <= 0:
        return None, None, None

    try:
        r = client.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            timeInForce="IOC",
            reduceOnly=False
        )
        order_id = (r.get("result",{}) or {}).get("orderId")
        return order_id, ref_price, qty
    except Exception as e:
        print("[ERROR] place_market:", e)
        return None, None, None

def place_reduce_market(symbol: str, exit_side: str, qty: Decimal) -> Optional[str]:
    try:
        r = client.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=exit_side,
            orderType="Market",
            qty=str(qty),
            timeInForce="IOC",
            reduceOnly=True,
            **_posidx()
        )
        return (r.get("result",{}) or {}).get("orderId")
    except Exception as e:
        print("[WARN] place_reduce_market:", e)
        return None

def cancel_all_orders(symbol: str):
    try:
        client.cancel_all_orders(category=CATEGORY, symbol=symbol)
    except Exception as e:
        print("[WARN] cancel_all_orders:", e)

# ===================== 상태/전역 =====================
OPEN_POS: Dict[str, dict] = {}    # symbol -> {...}
COOLDOWN_UNTIL: Dict[str, float] = {}

def _equity_now_price_map() -> Dict[str, float]:
    pm = {}
    for sym in OPEN_POS.keys():
        try:
            _, _, mid = get_quote(sym)
            pm[sym] = mid
        except Exception:
            pass
    return pm

def get_equity_breakdown():
    cash = 0.0  # (간단 표기)
    upnl_total = 0.0
    upnl_each = {}
    for sym, p in OPEN_POS.items():
        try:
            _, _, mid = get_quote(sym)
            qty = float(p.get("qty", 0.0) or 0.0)
            entry = float(p["entry"])
            side = p["side"]
            pnl = qty * ((mid - entry) if side=="Buy" else (entry - mid))
            upnl_each[sym] = pnl
            upnl_total += pnl
        except Exception:
            pass
    equity = cash + upnl_total
    return equity, cash, upnl_total, upnl_each

# ===================== 품질/포지션 크기 및 진입 =====================
CONF_MIN     = _envf("CONF_MIN", 0.75)
PNL_MIN_USD  = _envf("PNL_MIN_USD", 0.01)
SHARPE_MIN   = _envf("SHARPE_MIN", 0.15)
KELLY_CAP    = _envf("KELLY_CAP", 1.0)
KELLY_MAX_ON_CASH = _envf("KELLY_MAX_ON_CASH", 0.35)

def _phi(z: float)->float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def score_signal_with_kelly_sharpe(now_price: float, target_price: float, side: str,
                                   horizon_sec: int, est_qty: float, sigma_per_min: float, confidence: float)->dict:
    STOP_RATIO = 1.0
    sign = 1.0 if side=="Buy" else -1.0
    dist = (target_price - now_price) * sign
    exp_usd = max(0.0, dist) * float(est_qty)
    if dist <= 0:
        return {"ok": False, "reason": "target not favorable", "exp_usd": 0.0, "sharpe": 0.0, "kelly_frac": 0.0, "score": 0.0}

    horizon_min = max(1.0, horizon_sec / 60.0)
    exp_ret_pct = dist / max(now_price, 1e-9)
    sigma_h = max(1e-9, sigma_per_min * math.sqrt(horizon_min))
    sharpe = exp_ret_pct / sigma_h

    p_win = float(np.clip(_phi(sharpe), 0.0, 1.0))
    R = 1.0 / max(1e-9, STOP_RATIO)
    q = 1.0 - p_win
    kelly_raw = p_win - (q / R)
    kelly_frac = float(np.clip(kelly_raw, 0.0, KELLY_CAP))

    sharpe_norm = float(np.clip(sharpe / 1.0, 0.0, 1.0))
    pnl_norm = float(np.clip(exp_usd / max(1e-9, 5.0 * PNL_MIN_USD), 0.0, 1.0))
    score = 0.4*sharpe_norm + 0.4*float(np.clip(confidence,0.0,1.0)) + 0.2*pnl_norm

    if confidence < CONF_MIN:
        return {"ok": False, "reason": f"conf<{CONF_MIN}", "exp_usd": exp_usd, "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}
    if exp_usd < PNL_MIN_USD:
        return {"ok": False, "reason": f"pnl<{PNL_MIN_USD}$", "exp_usd": exp_usd, "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}
    if SHARPE_MIN is not None and sharpe < SHARPE_MIN:
        return {"ok": False, "reason": f"sharpe<{SHARPE_MIN}", "exp_usd": exp_usd, "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}

    return {"ok": True, "reason": "ok", "exp_usd": exp_usd,
            "sharpe": sharpe, "kelly_frac": kelly_frac, "score": score}

def _pick_side(base_side: str) -> str:
    if MODE == "model":
        return base_side
    if MODE == "inverse":
        return _opp(base_side)
    if MODE == "random":
        return random.choice(["Buy","Sell"])
    return base_side

def try_enter(symbol: str) -> bool:
    if symbol in OPEN_POS: return False
    if symbol in EXCLUDE_SYMBOLS: return False
    if not trading_allowed_now(): return False
    now = time.time()
    if COOLDOWN_UNTIL.get(symbol, 0) > now: return False
    if len(OPEN_POS) >= MAX_OPEN: return False

    # 모델 예측/모멘텀
    df = fetch_recent_data(symbol, 120)
    tp_pred, tsec, conf = predict_future_price(symbol, df)
    bid, ask, mid = get_quote(symbol)
    now_price = mid
    base_side = "Buy" if tp_pred > now_price else "Sell"
    side = _pick_side(base_side)

    # 스프레드/괴리 가드
    if bid > 0 and ask > 0 and mid > 0:
        spread_bps = (ask - bid) / mid * 10_000.0
        if spread_bps > SPREAD_REJECT_BPS:
            return False
    last_close = float(df["close"].iloc[-1])
    gap_bps = abs(now_price - last_close) / max(last_close, 1e-9) * 10_000.0
    if gap_bps > GAP_REJECT_BPS:
        return False

    # 리스크/현금 가정
    equity, cash, upnl, _ = get_equity_breakdown()
    equity = max(1.0, equity)
    use_cash_cap = equity * RISK_PCT_OF_EQUITY
    use_cash_base = min(use_cash_cap, equity * ENTRY_PORTION)
    if use_cash_base <= 0: return False

    # 품질 점수
    sigma_per_min = compute_sigma_pct_from_df(df, 120)
    target_price_for_score = now_price * (1.0 + (TP1_BPS/10000.0 if side == "Buy" else -TP1_BPS/10000.0))
    denom = now_price * (1.0 / LEVERAGE + TAKER_FEE)
    est_qty_float = (use_cash_base / denom) if denom > 0 else 0.0

    qm = score_signal_with_kelly_sharpe(now_price, target_price_for_score, side, int(tsec),
                                        est_qty_float, sigma_per_min, conf)
    if not qm["ok"]:
        return False
    kf = min(qm["kelly_frac"], KELLY_MAX_ON_CASH)
    use_final = use_cash_base * max(0.08, kf)

    # 진입
    oid, ref_px, qty = place_market_safely(symbol, side, float(use_final), LEVERAGE)
    if oid is None or ref_px is None or qty is None:
        return False

    entry = float(ref_px)
    qty = float(qty)

    # SL 초기
    if side == "Buy":
        sl_price = entry * (1.0 - STOP_LOSS_PCT)
    else:
        sl_price = entry * (1.0 + STOP_LOSS_PCT)

    # 티어별 목표가
    def tp_price_from_bps(bps: float)->float:
        return entry * (1.0 + bps/10000.0) if side=="Buy" else entry * (1.0 - bps/10000.0)
    tp1_price = tp_price_from_bps(TP1_BPS)
    tp2_price = tp_price_from_bps(TP2_BPS)
    tp3_price = tp_price_from_bps(TP3_BPS)

    OPEN_POS[symbol] = {
        "side": side,
        "qty": qty,                 # 추정 체결 수량
        "entry": entry,
        "entry_ts": time.time(),
        "sl_price": float(sl_price),
        "tp_prices": [tp1_price, tp2_price, tp3_price],
        "tp_done": [False, False, False],
        "tp_ratios": [TP1_RATIO, TP2_RATIO, 1.0],  # TP3는 남은 전량
        "be_moved": False,
        "peak": entry,
        "trough": entry,
        "mode": MODE,
    }

    _write_trade_row({
        "ts": int(time.time()), "event": "ENTRY", "symbol": symbol, "side": side,
        "qty": f"{qty:.10f}", "entry_price": f"{entry:.6f}", "tp_price": "", "sl_price": f"{float(sl_price):.6f}",
        "order_id": oid, "tp_order_id": "", "sl_order_id": "",
        "reason": "open", "pnl_usdt": "", "roi": "", "extra": f"tp1={tp1_price:.6f},tp2={tp2_price:.6f},tp3={tp3_price:.6f}", "mode": MODE,
    })
    return True

# ===================== 포지션 관리 =====================
def _close_reduce(symbol: str, side: str, qty: float, reason: str):
    exit_side = _opp(side)
    oid = place_reduce_market(symbol, exit_side, _d(qty))
    _, _, mid = get_quote(symbol)
    return oid, mid

def _maybe_close_all(symbol: str, reason: str):
    p = OPEN_POS.get(symbol)
    if not p: return
    side = p["side"]
    qty = float(p.get("qty", 0.0) or 0.0)
    if qty <= 0: qty = 1e9  # fallback: 전량 reduceOnly 유도
    oid, px = _close_reduce(symbol, side, qty, reason)
    pnl = qty * ((px - p["entry"]) if side=="Buy" else (p["entry"] - px))
    _write_trade_row({
        "ts": int(time.time()), "event": "EXIT_ALL", "symbol": symbol, "side": side,
        "qty": f"{qty:.10f}", "entry_price": f"{p['entry']:.6f}", "tp_price": "", "sl_price": f"{p['sl_price']:.6f}",
        "order_id": "", "tp_order_id": "", "sl_order_id": "",
        "reason": reason, "pnl_usdt": f"{pnl:.6f}", "roi": f"{_roi(side,p['entry'],px):.6f}", "extra": "", "mode": p.get("mode", MODE),
    })
    cancel_all_orders(symbol)
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
    OPEN_POS.pop(symbol, None)

def manage_positions():
    _log_equity_snapshot(get_equity_breakdown)
    now = time.time()

    for symbol, p in list(OPEN_POS.items()):
        side = p["side"]
        entry = float(p["entry"])
        qty_all = float(p.get("qty", 0.0) or 0.0)
        bid, ask, mid = get_quote(symbol)

        # 시간 제한
        held = now - p["entry_ts"]
        if held >= MAX_HOLD_SEC:
            _maybe_close_all(symbol, "EXIT_TIMESTOP")
            continue

        # 즉시익절 (절대/상대)
        notional = entry * qty_all
        pnl_usd = qty_all * ((mid - entry) if side=="Buy" else (entry - mid))
        if TAKE_PROFIT_USD > 0.0 and pnl_usd >= TAKE_PROFIT_USD:
            _maybe_close_all(symbol, "EXIT_TP_USD")
            continue
        if TAKE_PROFIT_REL_PCT > 0.0 and pnl_usd >= notional * TAKE_PROFIT_REL_PCT:
            _maybe_close_all(symbol, "EXIT_TP_REL")
            continue

        # 피크/트로프 갱신
        if side == "Buy":
            p["peak"] = max(p["peak"], mid)
        else:
            p["trough"] = min(p["trough"], mid)

        # 티어별 부분익절 체크
        for i, tp_px in enumerate(p["tp_prices"]):
            if p["tp_done"][i]:
                continue
            hit = (mid >= tp_px) if side == "Buy" else (mid <= tp_px)
            if hit:
                # 청산 수량 계산
                remaining_qty = float(p.get("qty", 0.0) or 0.0)
                if remaining_qty <= 0: break
                if i < 2:
                    close_qty = max(0.0, min(remaining_qty, remaining_qty * p["tp_ratios"][i]))
                else:
                    close_qty = remaining_qty  # TP3는 전량

                oid, px = _close_reduce(symbol, side, close_qty, f"TP{i+1}")
                # 수량 차감
                p["qty"] = max(0.0, remaining_qty - close_qty)
                p["tp_done"][i] = True

                _write_trade_row({
                    "ts": int(time.time()), "event": f"EXIT_TP{i+1}", "symbol": symbol, "side": side,
                    "qty": f"{close_qty:.10f}", "entry_price": f"{entry:.6f}", "tp_price": f"{tp_px:.6f}", "sl_price": f"{p['sl_price']:.6f}",
                    "order_id": "", "tp_order_id": oid or "", "sl_order_id": "",
                    "reason": f"TP{i+1}", "pnl_usdt": f"{close_qty*((px-entry) if side=='Buy' else (entry-px)):.6f}",
                    "roi": f"{_roi(side,entry,px):.6f}", "extra": "", "mode": p.get("mode", MODE),
                })

                # BE 이동: 설정된 티어 이후에만
                if not p["be_moved"] and (i+1) >= BE_AFTER_TIER:
                    eps = BE_EPS_BPS / 10_000.0
                    if side == "Buy":
                        p["sl_price"] = entry * (1.0 + eps)
                    else:
                        p["sl_price"] = entry * (1.0 - eps)
                    p["be_moved"] = True
                    _write_trade_row({
                        "ts": int(time.time()), "event": "MOVE_BE", "symbol": symbol, "side": side,
                        "qty": "", "entry_price": f"{entry:.6f}", "tp_price": "", "sl_price": f"{p['sl_price']:.6f}",
                        "order_id": "", "tp_order_id": "", "sl_order_id": "",
                        "reason": f"BE_AFTER_TIER_{BE_AFTER_TIER}", "pnl_usdt": "", "roi": "", "extra": f"eps_bps={BE_EPS_BPS}", "mode": p.get("mode", MODE),
                    })

        # 트레일링 SL: 특정 티어 완료 이후에만
        tiers_done = sum(1 for x in p["tp_done"] if x)
        if tiers_done >= TRAIL_AFTER_TIER:
            if side == "Buy":
                trail_px = p["peak"] * (1.0 - TRAIL_BPS / 10_000.0)
                p["sl_price"] = max(p["sl_price"], trail_px)
            else:
                trail_px = p["trough"] * (1.0 + TRAIL_BPS / 10_000.0)
                p["sl_price"] = min(p["sl_price"], trail_px)

        # SL 체결
        if (side == "Buy" and mid <= p["sl_price"]) or (side == "Sell" and mid >= p["sl_price"]):
            _maybe_close_all(symbol, "EXIT_SL")
            continue

        # 전량 청산됐으면 종료
        if float(p.get("qty", 0.0) or 0.0) <= 0.0:
            cancel_all_orders(symbol)
            COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
            OPEN_POS.pop(symbol, None)

# ===================== 메인 루프 =====================
def main_loop():
    print(f"[RUN] MODE={MODE} TESTNET={TESTNET} LEVERAGE={LEVERAGE} USE_DEEPAR={USE_DEEPAR}")
    while True:
        try:
            # 엔트리 시도
            for sym in SYMBOLS:
                try_enter(sym)
            # 포지션 관리
            manage_positions()
        except KeyboardInterrupt:
            print("Interrupted. Exiting...")
            break
        except Exception as e:
            print("[ERR] main:", e)
        time.sleep(1.0)

if __name__ == "__main__":
    main_loop()
