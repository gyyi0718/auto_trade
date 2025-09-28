# -*- coding: utf-8 -*-
# PAPER: TCN + CatBoost 조합 (DeepAR 제거)
import os, time, math, csv, random, threading, collections
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pybit.unified_trading import HTTP
from concurrent.futures import ThreadPoolExecutor
import certifi, json

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import warnings, logging
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ===== ENV =====
CATEGORY = "linear"
INTERVAL = "1"  # 1m

SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,LINEAUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")

TIMEOUT_MODE = os.getenv("TIMEOUT_MODE", "model")   # fixed|atr|model
VOL_WIN = int(os.getenv("VOL_WIN", "60"))
TIMEOUT_K = float(os.getenv("TIMEOUT_K", "1.5"))
TIMEOUT_MIN = int(os.getenv("TIMEOUT_MIN", "1"))
TIMEOUT_MAX = int(os.getenv("TIMEOUT_MAX", "3"))
PRED_HORIZ_MIN = int(os.getenv("PRED_HORIZ_MIN","1"))
_v = os.getenv("TIMEOUT_TARGET_BPS")
TIMEOUT_TARGET_BPS = float(_v) if (_v is not None and _v.strip() != "") else None

ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model|inverse|random

START_EQUITY = float(os.getenv("START_EQUITY","1000.0"))
LEVERAGE = float(os.getenv("LEVERAGE","20"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.5"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","0.5"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))

TP1_BPS = float(os.getenv("TP1_BPS","50.0"))
TP2_BPS = float(os.getenv("TP2_BPS","80.0"))
TP3_BPS = float(os.getenv("TP3_BPS","172.0"))
SL_ROI_PCT = float(os.getenv("SL_ROI_PCT","0.009"))
TP1_RATIO = float(os.getenv("TP1_RATIO","0.2"))
TP2_RATIO = float(os.getenv("TP2_RATIO","0.3"))
BE_EPS_BPS = float(os.getenv("BE_EPS_BPS","2.0"))
TRAIL_BPS = float(os.getenv("TRAIL_BPS","50.0"))
TRAIL_AFTER_TIER = int(os.getenv("TRAIL_AFTER_TIER","2"))

ABS_TP_USD = float(os.getenv("ABS_TP_USD","0.0"))
ABS_K = float(os.getenv("ABS_K","1.0"))
ABS_TP_USD_FLOOR = float(os.getenv("ABS_TP_USD_FLOOR","1.0"))

N_CONSEC_SIGNALS = int(os.getenv("N_CONSEC_SIGNALS","2"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","30"))
SESSION_MAX_TRADES = int(os.getenv("SESSION_MAX_TRADES","1000"))
SESSION_MAX_MINUTES = int(os.getenv("SESSION_MAX_MINUTES","5"))

SCAN_PARALLEL = str(os.getenv("SCAN_PARALLEL","1")).lower() in ("1","y","yes","true")
SCAN_WORKERS  = int(os.getenv("SCAN_WORKERS","8"))

TRADES_CSV = os.getenv("TRADES_CSV",f"paper_trades_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_CSV = os.getenv("EQUITY_CSV",f"paper_equity_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
EQUITY_MTM_CSV = os.getenv("EQUITY_MTM_CSV", f"paper_equity_mtm_{ENTRY_MODE}_{START_EQUITY}_{LEVERAGE}.csv")
TRADES_FIELDS = ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","reason","mode"]
EQUITY_FIELDS = ["ts","equity"]
VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")

# === META PAUSE ===
META_ON = int(os.getenv("META_ON","1"))
META_WIN = int(os.getenv("META_WIN","30"))
META_MIN_HIT = float(os.getenv("META_MIN_HIT","0.5"))
META_MIN_EVR = float(os.getenv("META_MIN_EVR","0.0"))
META_MAX_DD_BPS = float(os.getenv("META_MAX_DD_BPS","-300.0"))
META_COOLDOWN = int(os.getenv("META_COOLDOWN","60"))

from collections import deque
META_TRADES: Dict[str, deque] = {s: deque(maxlen=max(5, META_WIN)) for s in SYMBOLS}
META_BLOCK_UNTIL: Dict[str, float] = {}

# ===== STATE =====
_equity_cache = START_EQUITY
POSITION_DEADLINE: Dict[str,int] = {}
STATE: Dict[str,dict] = {}
ENTRY_LOCK = threading.Lock()
ENTRY_HISTORY: Dict[str,collections.deque] = {s: collections.deque(maxlen=max(1,N_CONSEC_SIGNALS)) for s in SYMBOLS}
COOLDOWN_UNTIL: Dict[str,float] = {}
SESSION_START_TS = time.time()
SESSION_TRADES = 0

# ===== MARKET (public only) =====
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

def _bps(pnl_usd: float, notional_usd: float) -> float:
    if notional_usd <= 0: return 0.0
    return (pnl_usd / notional_usd) * 10_000.0

def _meta_push(symbol: str, pnl_bps: float):
    META_TRADES.setdefault(symbol, deque(maxlen=max(5, META_WIN))).append(float(pnl_bps))

def _meta_stats(symbol: str) -> Tuple[float,float,float]:
    arr = list(META_TRADES.get(symbol, []))
    if not arr: return 1.0, 0.0, 0.0
    n = len(arr)
    hit = sum(1 for x in arr if x > 0) / n
    mean = sum(arr)/n
    peak = -1e18; mdd = 0.0; cum = 0.0
    for x in arr:
        cum += x
        if cum > peak: peak = cum
        mdd = min(mdd, cum - peak)
    return hit, mean, mdd

# ===== Market helpers =====
def get_quote(symbol: str) -> Tuple[float,float,float]:
    r = mkt.get_tickers(category=CATEGORY, symbol=symbol)
    row = ((r.get("result") or {}).get("list") or [{}])[0]
    bid = float(row.get("bid1Price") or 0.0)
    ask = float(row.get("ask1Price") or 0.0)
    last = float(row.get("lastPrice") or 0.0)
    mid = (bid+ask)/2.0 if bid>0 and ask>0 else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

def get_recent_1m(symbol: str, minutes: int = 200) -> Optional[pd.DataFrame]:
    r = mkt.get_kline(category=CATEGORY, symbol=symbol, interval=INTERVAL, limit=min(minutes,1000))
    rows = (r.get("result") or {}).get("list") or []
    if not rows: return None
    rows = rows[::-1]
    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    return df

def recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 5:
        return 10.0
    logret = np.log(df["close"]).diff().abs()
    v_bps = float(logret.median() * 1e4)
    return max(v_bps, 1.0)

def fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

def tp_from_bps(entry: float, bps: float, side: str) -> float:
    return entry*(1.0 + bps/10000.0) if side=="Buy" else entry*(1.0 - bps/10000.0)

def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

def _entry_cost_bps()->float:
    return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def _exit_cost_bps()->float:
    return TAKER_FEE*1e4 + SLIPPAGE_BPS_TAKER

def _log_thr_from_bps(target_bps: float) -> float:
    return math.log1p(target_bps / 1e4)

def calc_timeout_minutes(symbol: str, side: Optional[str] = None) -> int:
    if TIMEOUT_MODE == "fixed":
        return max(1, int(PRED_HORIZ_MIN))
    if TIMEOUT_MODE == "model":
        target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
        m2 = tcn_time_to_target_minutes(symbol, side or "Buy", target_bps)
        if m2 is not None:
            chosen = int(min(max(m2, TIMEOUT_MIN), TIMEOUT_MAX))
            if VERBOSE: print(f"[TIMEOUT][{symbol}] tcn={m2}m -> {chosen}m (target={target_bps}bps)")
            return chosen
        if VERBOSE: print(f"[TIMEOUT][{symbol}] model-time unavailable -> fallback to ATR")
    target_bps = TIMEOUT_TARGET_BPS if TIMEOUT_TARGET_BPS is not None else TP1_BPS
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    need = math.ceil((target_bps / v_bps) * TIMEOUT_K)
    return int(min(max(need, TIMEOUT_MIN), TIMEOUT_MAX))

def _dynamic_abs_tp_usd(symbol:str, mid:float, qty:float)->float:
    if ABS_TP_USD > 0:
        return ABS_TP_USD
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    atr_usd = (v_bps/1e4) * mid * qty
    return max(ABS_TP_USD_FLOOR, ABS_K * atr_usd)

# ===== TCN (binary-threshold driver) =====
TCN_CKPT = os.getenv("TCN_CKPT", "tcn_best.pt")
TCN_FEATS = ["ret","rv","mom","vz"]
TCN_SEQ_LEN_FALLBACK = 240

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

def weight_norm_conv(in_c, out_c, k, dilation):
    pad = (k-1)*dilation
    return nn.utils.weight_norm(nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, dilation=dilation))

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation, dropout):
        super().__init__()
        self.conv1 = weight_norm_conv(in_c, out_c, k, dilation)
        self.chomp1 = Chomp1d((k-1)*dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm_conv(out_c, out_c, k, dilation)
        self.chomp2 = Chomp1d((k-1)*dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBinary(nn.Module):
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, out_dim=2):
        super().__init__()
        layers=[]; ch_in=in_feat
        for i in range(levels):
            layers += [TemporalBlock(ch_in, hidden, k, 2**i, dropout)]
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)  # out_dim=1(cls), 2(joint)

    def forward(self, x):
        x = x.transpose(1,2)
        h = self.tcn(x)[:, :, -1]
        o = self.head(h)
        if o.shape[-1] == 1:
            return None, o.squeeze(-1)            # (mu=None, logit)
        else:
            return o[:,0], o[:,1]                 # (mu, logit)


def _build_tcn_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("timestamp").copy()
    c = g["close"].astype(float).values
    v = g["volume"].astype(float).values
    ret = np.zeros_like(c, dtype=np.float64)
    ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))
    win = 60
    rv = pd.Series(ret).rolling(win, min_periods=win//3).std().bfill().values
    ema_win=60; alpha=2/(ema_win+1)
    ema=np.zeros_like(c, dtype=np.float64); ema[0]=c[0]
    for i in range(1,len(c)): ema[i] = alpha*c[i] + (1-alpha)*ema[i-1]
    mom = (c/np.maximum(ema,1e-12) - 1.0)
    v_ma = pd.Series(v).rolling(win, min_periods=win//3).mean().bfill().values
    v_sd = pd.Series(v).rolling(win, min_periods=win//3).std().replace(0, np.nan).bfill().values
    vz  = (v - v_ma) / np.where(v_sd>0, v_sd, 1.0)
    return pd.DataFrame({"timestamp": g["timestamp"].values,
                         "ret": ret, "rv": rv, "mom": mom, "vz": vz})

def _apply_tcn_scaler(X: np.ndarray, mu, sd):
    if mu is None or sd is None: return X
    mu = np.asarray(mu, dtype=np.float32); sd = np.asarray(sd, dtype=np.float32)
    sd = np.where(sd>0, sd, 1.0)
    return (X - mu) / sd

TCN_MODEL = None; TCN_CFG={}; TCN_MU=None; TCN_SD=None
TCN_HAS_MU = False
TCN_THR_PROB = float(os.getenv("TCN_THR_PROB","0.55"))  # cls 모드용 확률 임계

try:
    _ckpt = torch.load(TCN_CKPT, map_location="cpu", weights_only=True)
    TCN_CFG = _ckpt.get("cfg", {})
    feats   = TCN_CFG.get("FEATS", TCN_FEATS)
    in_feat = len(feats)
    hidden  = int(TCN_CFG.get("HIDDEN", 64))
    levels  = int(TCN_CFG.get("LEVELS", 5))
    k       = int(TCN_CFG.get("K", 3))
    dropout = float(TCN_CFG.get("DROPOUT", 0.3))
    loss_mode = (TCN_CFG.get("LOSS_MODE","cls") or "cls").lower()
    out_dim = 1 if loss_mode=="cls" else 2

    TCN_MODEL = TCNBinary(in_feat=in_feat, hidden=hidden, levels=levels, k=k,
                          dropout=dropout, out_dim=out_dim).eval()
    TCN_MODEL.load_state_dict(_ckpt["model"], strict=True)

    TCN_MU = _ckpt.get("scaler_mu", None)
    TCN_SD = _ckpt.get("scaler_sd", None)
    TCN_FEATS = feats
    TCN_SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))
    TCN_HAS_MU = (out_dim == 2)
except Exception as e:
    TCN_MODEL = None
    TCN_SEQ_LEN = TCN_SEQ_LEN_FALLBACK
    print(f"[WARN] TCN load failed: {e}")

def _tcn_thr():
    mode = os.getenv("THR_MODE", "fixed").lower()
    if mode == "fee" and TCN_HAS_MU:
        return float(fee_threshold())
    # μ 임계(bps→ratio), cls모드에선 쓰지 않음
    return float(os.getenv("MODEL_THR_BPS", "5.0")) / 1e4

@torch.no_grad()
def tcn_direction(symbol: str):
    if TCN_MODEL is None:
        return (None, 0.0, _tcn_thr(), 0.0)

    df = get_recent_1m(symbol, minutes=TCN_SEQ_LEN + 10)
    if df is None or len(df) < (TCN_SEQ_LEN+1):
        return (None, 0.0, _tcn_thr(), 0.0)

    feats = _build_tcn_features(df)
    X = feats[TCN_FEATS].tail(TCN_SEQ_LEN).to_numpy().astype(np.float32)
    X = _apply_tcn_scaler(X, TCN_MU, TCN_SD)
    x_t = torch.from_numpy(X[None, ...])

    mu, logit = TCN_MODEL(x_t)
    thr_mu = _tcn_thr()

    if TCN_HAS_MU:
        mu = float(mu.item()); lg = float(logit.item())
        if   mu >  thr_mu: side = "Buy"
        elif mu < -thr_mu: side = "Sell"
        else:              side = None
        if VERBOSE: print(f"[DEBUG][TCN][{symbol}] mu={mu:.6g} thr={thr_mu:.6g} logit={lg:.6g} -> {side}")
        return side, mu, thr_mu, lg
    else:
        lg = float(logit.item())
        p  = 1.0 / (1.0 + math.exp(-lg))
        buy_thr  = TCN_THR_PROB
        sell_thr = 1.0 - TCN_THR_PROB
        if   p >= buy_thr:  side = "Buy"
        elif p <= sell_thr: side = "Sell"
        else:               side = None
        if VERBOSE: print(f"[DEBUG][TCN][{symbol}] p={p:.3f} thr={TCN_THR_PROB:.2f} -> {side}")
        return side, 0.0, 0.0, lg  # mu 없음

@torch.no_grad()
def tcn_time_to_target_minutes(symbol: str, side: str, target_bps: float):
    # μ 없는(cls) 모델이면 모델 기반 타임아웃 비활성화 → None 반환(ATR로 폴백)
    if not TCN_HAS_MU:
        return None
    t_side, mu_t, _thr, _ = tcn_direction(symbol)
    if mu_t is None:
        return None
    step = abs(float(mu_t))
    if step <= 1e-8:
        return None
    thr = math.log1p(target_bps / 1e4)
    need = int(math.ceil(thr / step))
    return max(1, need)

# ===== CatBoost (meta) =====
from catboost import CatBoostClassifier, Pool

CATBOOST_CBM  = os.getenv("CATBOOST_CBM",  "../multimodel/models/catboost_meta.cbm")
CATBOOST_META = os.getenv("CATBOOST_META", "../multimodel/models/catboost_meta_features.json")

CB_MODEL: Optional[CatBoostClassifier] = None
CB_META: dict = {}
try:
    CB_MODEL = CatBoostClassifier()
    CB_MODEL.load_model(CATBOOST_CBM)
    with open(CATBOOST_META, "r", encoding="utf-8") as f:
        CB_META = json.load(f)
except Exception as e:
    CB_MODEL = None
    CB_META = {}
    print(f"[WARN] CatBoost load failed: {e}")

def _latest_base_feats(df: pd.DataFrame) -> Optional[dict]:
    # 요구 최소 길이
    need = 120
    if df is None or len(df) < need: return None
    g = df.sort_values("timestamp").copy()
    c = g["close"].astype(float).values
    h = g["high"].astype(float).values
    l = g["low"].astype(float).values
    v = g["volume"].astype(float).values
    logc = np.log(np.maximum(c, 1e-12))
    ret1 = np.zeros_like(logc); ret1[1:] = logc[1:] - logc[:-1]
    def roll_std(arr, w):
        s = pd.Series(arr).rolling(w, min_periods=max(2, w//3)).std()
        return float(s.iloc[-1])
    def mom_w(arr, w):
        ema = pd.Series(arr).ewm(span=w, adjust=False).mean()
        return float(arr[-1]/max(ema.iloc[-1], 1e-12) - 1.0)
    def volz(arr, w):
        s = pd.Series(arr)
        mu = s.rolling(w, min_periods=max(2,w//3)).mean()
        sd = s.rolling(w, min_periods=max(2,w//3)).std().replace(0, np.nan)
        val = (arr[-1] - float(mu.iloc[-1])) / (float(sd.iloc[-1]) if pd.notna(sd.iloc[-1]) and sd.iloc[-1]>0 else 1.0)
        return float(val)
    # ATR14
    prev_close = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[
        (h-l),
        np.abs(h - prev_close),
        np.abs(l - prev_close)
    ], axis=1)
    atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().iloc[-1]
    feats = {
        "ret1": float(ret1[-1]),
        "rv5":  roll_std(ret1, 5),  "rv15": roll_std(ret1, 15),
        "rv30": roll_std(ret1, 30), "rv60": roll_std(ret1, 60),
        "mom5":  mom_w(c, 5),   "mom15": mom_w(c, 15),
        "mom30": mom_w(c, 30),  "mom60": mom_w(c, 60),
        "vz20": volz(v, 20), "vz60": volz(v, 60),
        "atr14": float(atr14 if pd.notna(atr14) else 0.0),
    }
    if any([not np.isfinite(v) for v in feats.values()]): return None
    return feats

@torch.no_grad()
def catboost_direction(symbol: str) -> Tuple[Optional[str], float]:
    """
    TCN+CatBoost 합의:
    - TCN 방향(side) 존재 && |mu|>=thr
    - CatBoost 확률 최고클래스가 Long/Short 이고 TCN과 부호 일치
    - 확률 threshold 이상
    """
    if CB_MODEL is None or not CB_META:
        return (None, 0.0)

    # 최신 데이터 수집 길이: TCN길이+60
    minutes = max(300, int(TCN_SEQ_LEN + 80))
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < (TCN_SEQ_LEN + 61):
        return (None, 0.0)

    # TCN
    t_side, t_mu, t_thr, t_logit = tcn_direction(symbol)
    if t_side is None or abs(t_mu) < t_thr:
        return (None, 0.0)

    # CatBoost 입력 구성
    base = _latest_base_feats(df)
    if base is None:
        return (None, 0.0)

    feat_cols: List[str] = CB_META.get("feature_cols", [])
    cat_cols:  List[str] = CB_META.get("cat_cols", ["symbol"])
    thr_prob:  float = float(CB_META.get("threshold_prob", 0.55))

    # TCN 특성이 요구되면 채워주기
    row = dict(base)
    if "tcn_mu" in feat_cols:    row["tcn_mu"] = float(t_mu)
    if "tcn_logit" in feat_cols: row["tcn_logit"] = float(t_logit)

    # 모든 필요 피처 보유 확인
    if not set(feat_cols).issubset(row.keys()):
        return (None, 0.0)

    X = pd.DataFrame([[row[c] for c in feat_cols] + [symbol] * len(cat_cols)],
                     columns=feat_cols + cat_cols)
    pool = Pool(X, cat_features=[len(feat_cols)+i for i in range(len(cat_cols))])

    proba = CB_MODEL.predict_proba(pool)[0]  # [P(class0), P(class1), P(class2)]
    pred_idx = int(np.argmax(proba))
    # train mapping: {-1:0, 0:1, +1:2}
    inv = {0:-1, 1:0, 2:+1}
    y_hat = inv.get(pred_idx, 0)

    if y_hat == 0:      # NoTrade
        return (None, float(np.max(proba)))
    if np.max(proba) < thr_prob:
        return (None, float(np.max(proba)))

    cb_side = "Buy" if y_hat > 0 else "Sell"

    # TCN 방향과 일치 필요
    if cb_side != t_side:
        return (None, float(np.max(proba)))

    # ENTRY_MODE 적용
    side = cb_side
    if ENTRY_MODE == "inverse":
        side = "Sell" if cb_side=="Buy" else "Buy"
    elif ENTRY_MODE == "random":
        side = side  # 그대로

    return side, float(np.max(proba))

# ===== Paper Broker =====
def _round_down(x: float, step: float) -> float:
    if step<=0: return x
    return math.floor(x/step)*step

class PaperBroker:
    def __init__(self):
        self.pos: Dict[str,dict] = {}
        self.qty_step = 0.001
        self.min_qty = 0.001
    def try_open(self, symbol: str, side: str, mid: float, ts: int):
        eq = get_wallet_equity()
        qty = _round_down(eq * ENTRY_EQUITY_PCT * LEVERAGE / max(mid,1e-9), self.qty_step)
        if qty < self.min_qty: return False
        self.pos[symbol] = {"side": side, "entry": mid, "qty": qty, "opened": ts}
        return True
    def try_close(self, symbol: str, ts: int) -> Optional[float]:
        if symbol not in self.pos: return None
        _,_,mark = get_quote(symbol)
        p = self.pos[symbol]; entry=p["entry"]; qty=p["qty"]
        pnl = (mark-entry)*qty if p["side"]=="Buy" else (entry-mark)*qty
        del self.pos[symbol]
        return pnl
    def positions(self): return self.pos

BROKER = PaperBroker()

def unrealized_pnl_usd(side: str, qty: float, entry: float, mark: float) -> float:
    return (mark - entry) * qty if side=="Buy" else (entry - mark) * qty

MIN_EXPECTED_ROI_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS","35.0"))
V_BPS_FLOOR = float(os.getenv("V_BPS_FLOOR","5.0"))

def _entry_allowed_by_filters(symbol:str, side:str, entry_ref:float)->bool:
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    if v_bps < V_BPS_FLOOR:
        if VERBOSE: print(f"[SKIP] {symbol} vol {v_bps:.1f}bps < floor {V_BPS_FLOOR}bps")
        return False
    tp1_px = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)
    need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS
    if edge_bps < need_bps:
        if VERBOSE: print(f"[SKIP] {symbol} ROI edge {edge_bps:.1f} < need {need_bps:.1f}bps")
        return False
    return True

# ===== I/O =====
def _ensure_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(TRADES_FIELDS)
    if not os.path.exists(EQUITY_CSV):
        with open(EQUITY_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)
    if not os.path.exists(EQUITY_MTM_CSV):
        with open(EQUITY_MTM_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(EQUITY_FIELDS)

def _log_equity_mtm(eq: float):
    _ensure_csv()
    with open(EQUITY_MTM_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{float(eq):.6f}"])
    if VERBOSE: print(f"[EQUITY_MTM] {float(eq):.2f}")

def _log_trade(row: dict):
    _ensure_csv()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(k,"") for k in TRADES_FIELDS])
    if VERBOSE:
        print(f"[TRADE] {row}")

def _log_equity(eq: float):
    global _equity_cache
    _equity_cache = float(eq)
    _ensure_csv()
    with open(EQUITY_CSV,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), f"{_equity_cache:.6f}"])
    if VERBOSE: print(f"[EQUITY] {_equity_cache:.2f}")

def get_equity_mtm() -> float:
    base = get_wallet_equity()
    upnl = 0.0
    for sym, p in BROKER.positions().items():
        side, qty, entry = p["side"], p["qty"], p["entry"]
        _,_,mark = get_quote(sym)
        if mark > 0:
            upnl += unrealized_pnl_usd(side, qty, entry, mark)
    return base + upnl

def get_wallet_equity() -> float:
    return float(_equity_cache)

def apply_trade_pnl(side: str, entry: float, exit: float, qty: float, taker_fee: float = TAKER_FEE):
    gross = (exit - entry) * qty if side=="Buy" else (entry - exit) * qty
    fee = (entry * qty + exit * qty) * taker_fee
    new_eq = get_wallet_equity() + gross - fee
    _log_equity(new_eq)
    return gross, fee, new_eq

# ===== Entry/Exit =====
def _meta_allows(symbol: str) -> bool:
    if not META_ON:
        return True
    if META_BLOCK_UNTIL.get(symbol, 0.0) > time.time():
        if VERBOSE: print(f"[META][BLOCK] {symbol} cooldown")
        return False
    hit, mean_bps, mdd = _meta_stats(symbol)
    ok = (hit >= META_MIN_HIT) and (mean_bps >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
    if not ok:
        META_BLOCK_UNTIL[symbol] = time.time() + META_COOLDOWN
        if VERBOSE:
            print(f"[META][PAUSE] {symbol} hit={hit:.2f} evr={mean_bps:.1f} mdd={mdd:.1f} -> rest {META_COOLDOWN}s")
        return False
    return True

def choose_entry(symbol: str):
    side, conf = catboost_direction(symbol)
    return side, conf

def try_enter(symbol: str):
    global SESSION_START_TS, SESSION_TRADES
    if time.time() - SESSION_START_TS >= SESSION_MAX_MINUTES * 60:
        SESSION_START_TS = time.time()
        SESSION_TRADES = 0
    if SESSION_MAX_TRADES > 0 and SESSION_TRADES >= SESSION_MAX_TRADES:
        if VERBOSE:
            remain = int(SESSION_MAX_MINUTES * 60 - (time.time() - SESSION_START_TS))
            print(f"[SKIP] session trade cap hit (remain {max(remain, 0)}s)")
        return
    if (time.time() - SESSION_START_TS) >= (SESSION_MAX_MINUTES * 60):
        if VERBOSE: print("[SKIP] session time cap hit")
        return
    if COOLDOWN_UNTIL.get(symbol, 0.0) > time.time():
        return
    if symbol in BROKER.positions():
        return

    side, conf = choose_entry(symbol)
    if side not in ("Buy", "Sell"):
        return
    if not _meta_allows(symbol):
        return
    dq = ENTRY_HISTORY[symbol]
    need = max(1, N_CONSEC_SIGNALS)
    dq.append(side)
    if need > 1:
        if len(dq) < need or any(x != side for x in dq):
            if VERBOSE: print(f"[SKIP] consec fail {symbol} need={need} got={list(dq)}")
            return

    bid, ask, mid = get_quote(symbol)
    if mid <= 0:
        return

    entry_ref = float(ask if side == "Buy" else bid)
    if not _entry_allowed_by_filters(symbol, side, entry_ref):
        return

    ts = int(time.time())
    with ENTRY_LOCK:
        if symbol in BROKER.positions():
            return
        if not BROKER.try_open(symbol, side, mid, ts):
            return

        horizon_min = max(1, int(calc_timeout_minutes(symbol, side)))
        POSITION_DEADLINE[symbol] = ts + horizon_min * 60

        tp1 = tp_from_bps(mid, TP1_BPS, side)
        tp2 = tp_from_bps(mid, TP2_BPS, side)
        tp3 = tp_from_bps(mid, TP3_BPS, side)
        sl  = mid * (1.0 - SL_ROI_PCT) if side == "Buy" else mid * (1.0 + SL_ROI_PCT)

        pos_qty = BROKER.positions()[symbol]["qty"]
        STATE[symbol] = {
            "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
            "tp1_filled": False, "tp2_filled": False, "be_moved": False,
            "abs_usd": _dynamic_abs_tp_usd(symbol, mid, pos_qty)
        }

        _log_trade({
            "ts": ts, "event": "ENTRY", "symbol": symbol, "side": side,
            "qty": f"{pos_qty:.8f}", "entry": f"{mid:.6f}", "exit": "",
            "tp1": f"{tp1:.6f}", "tp2": f"{tp2:.6f}", "tp3": f"{tp3:.6f}",
            "sl": f"{sl:.6f}", "reason": "open", "mode": "paper"
        })

        SESSION_TRADES += 1
        if VERBOSE:
            print(f"[OPEN] {symbol} {side} entry={mid:.4f} timeout={horizon_min}m "
                  f"abs≈{STATE[symbol]['abs_usd']:.2f}USD  [EQUITY]={get_wallet_equity():.2f}")

def close_and_log(symbol: str, reason: str):
    now = int(time.time())
    if symbol not in BROKER.positions(): return
    p = BROKER.positions()[symbol]
    side = p["side"]; entry = p["entry"]; qty = p["qty"]
    _,_,mark = get_quote(symbol)
    BROKER.try_close(symbol, now)
    gross, fee, new_eq = apply_trade_pnl(side, entry, mark, qty, taker_fee=TAKER_FEE)
    notional = max(entry * qty, 1e-9)
    pnl_bps = _bps(gross - fee, notional)
    _meta_push(symbol, pnl_bps)
    _log_trade({"ts": now, "event":"EXIT", "symbol":symbol, "side":"Sell" if side=="Buy" else "Buy",
                "qty":f"{qty:.8f}","entry":f"{entry:.6f}","exit":f"{mark:.6f}",
                "tp1":"","tp2":"","tp3":"","sl":"","reason":reason,"mode":"paper"})
    if VERBOSE:
        print(f"[{reason}] {symbol} gross={gross:.6f} fee={fee:.6f} eq={new_eq:.2f}")
    POSITION_DEADLINE.pop(symbol, None)
    STATE.pop(symbol, None)
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC

# ===== Monitor =====
def monitor_loop(poll_sec: float = 1.0, max_loops: int = 2):
    loops = 0
    while loops < max_loops:
        _log_equity_mtm(get_equity_mtm())
        now = int(time.time())
        for sym, p in list(BROKER.positions().items()):
            side=p["side"]; entry=p["entry"]; qty=p["qty"]
            _,_,mark = get_quote(sym)
            if mark<=0: continue

            st = STATE.get(sym, None)
            if st is None:
                tp1 = tp_from_bps(entry, TP1_BPS, side)
                tp2 = tp_from_bps(entry, TP2_BPS, side)
                tp3 = tp_from_bps(entry, TP3_BPS, side)
                sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
                st = {"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,
                      "tp1_filled":False,"tp2_filled":False,"be_moved":False,
                      "abs_usd": _dynamic_abs_tp_usd(sym, entry, qty)}
                STATE[sym]=st

            if (not st["tp1_filled"]) and ((mark>=st["tp1"] and side=="Buy") or (mark<=st["tp1"] and side=="Sell")):
                part = qty*TP1_RATIO
                apply_trade_pnl(side, entry, st["tp1"], part, taker_fee=TAKER_FEE)
                p["qty"] = qty = max(0.0, qty - part)
                st["tp1_filled"]=True
                _log_trade({"ts":int(time.time()),"event":"TP1_FILL","symbol":sym,"side":side,
                            "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{st['tp1']:.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})

            if qty>0 and (not st["tp2_filled"]) and ((mark>=st["tp2"] and side=="Buy") or (mark<=st["tp2"] and side=="Sell")):
                remain_ratio = 1.0 - TP1_RATIO
                part = min(qty, remain_ratio*TP2_RATIO)
                apply_trade_pnl(side, entry, st["tp2"], part, taker_fee=TAKER_FEE)
                p["qty"] = qty = max(0.0, qty - part)
                st["tp2_filled"]=True
                _log_trade({"ts":int(time.time()),"event":"TP2_FILL","symbol":sym,"side":side,
                            "qty":f"{part:.8f}","entry":f"{entry:.6f}","exit":f"{st['tp2']:.6f}",
                            "tp1":"","tp2":"","tp3":"","sl":"","reason":"","mode":"paper"})

            if st["tp2_filled"] and not st["be_moved"]:
                be_px = entry*(1.0 + BE_EPS_BPS/10000.0) if side=="Buy" else entry*(1.0 - BE_EPS_BPS/10000.0)
                st["sl"] = be_px
                st["be_moved"]=True
                _log_trade({"ts":int(time.time()),"event":"MOVE_SL","symbol":sym,"side":side,
                            "qty":"","entry":"","exit":"","tp1":"","tp2":"","tp3":"",
                            "sl":f"{be_px:.6f}","reason":"BE","mode":"paper"})

            if (st["tp1_filled"] if TRAIL_AFTER_TIER==1 else st["tp2_filled"]):
                if side=="Buy":
                    st["sl"] = max(st["sl"], mark*(1.0 - TRAIL_BPS/10000.0))
                else:
                    st["sl"] = min(st["sl"], mark*(1.0 + TRAIL_BPS/10000.0))

            if qty > 0:
                pnl = unrealized_pnl_usd(side, qty, entry, mark)
                abs_hit = pnl >= st["abs_usd"]
                sl_hit  = (mark <= st["sl"] and side=="Buy") or (mark >= st["sl"] and side=="Sell")
                ddl = POSITION_DEADLINE.get(sym)
                to_hit = (ddl and now >= ddl)
                if abs_hit or sl_hit or to_hit:
                    reason = "ABS_TP" if abs_hit else ("SL" if sl_hit else "HORIZON_TIMEOUT")
                    close_and_log(sym, reason)
                    continue

        loops += 1
        time.sleep(poll_sec)

# ===== Main =====
def main():
    print(f"[START] PAPER TCN+CatBoost | TESTNET={TESTNET}")
    _log_equity(get_wallet_equity())
    _log_equity_mtm(get_equity_mtm())
    while True:
        if SCAN_PARALLEL:
            with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
                ex.map(try_enter, SYMBOLS)
        else:
            for s in SYMBOLS:
                try_enter(s)
        monitor_loop(1.0, 2)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
