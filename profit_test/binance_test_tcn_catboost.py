# -*- coding: utf-8 -*-
# PAPER: TCN + CatBoost 조합 (Binance USDT-M 선물 시세로 페이퍼 테스트)
import os, time, math, csv, random, threading, collections
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import certifi, json, requests

# ====== TLS certs (윈도우 이슈 방지) ======
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import warnings, logging
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ===== ENV =====
CATEGORY = "linear"
INTERVAL = os.getenv("INTERVAL", "1")  # "1" (분) → 바이낸스 "1m"로 매핑
def _bx_interval(s: str) -> str:
    s = str(s).strip().lower()
    if s.endswith("m"): return s
    return f"{s}m"

SYMBOLS = os.getenv("SYMBOLS","ORDERUSDT,FORMUSDT,ASTERUSDT,AIAUSDT,0GUSDT,STBLUSDT,LINEAUSDT,BARDUSDT,SOMIUSDT,UBUSDT,OPENUSDT").split(",")
DIAG = str(os.getenv("DIAG","0")).lower() in ("1","y","yes","true")
TRACE = str(os.getenv("TRACE","1")).lower() in ("1","y","yes","true")

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
LEVERAGE = float(os.getenv("LEVERAGE","25"))
ENTRY_EQUITY_PCT = float(os.getenv("ENTRY_EQUITY_PCT","0.5"))
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","0.5"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))

TP1_BPS = float(os.getenv("TP1_BPS","50.0"))
TP2_BPS = float(os.getenv("TP2_BPS","80.0"))
TP3_BPS = float(os.getenv("TP3_BPS","150.0"))
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

# ==== PATCH: globals (paste near other globals) ====
from collections import deque

# Dynamic CB threshold (quantile-based)
CB_Q_WIN   = int(os.getenv("CB_Q_WIN", "200"))    # window size for recent confidences
CB_Q_PERC  = float(os.getenv("CB_Q_PERC", "0.70"))# use 70th percentile as dynamic cut
MIN_CB_THR = float(os.getenv("MIN_CB_THR", "0.55"))

# ROI filters (dynamic boosts)
ROI_BASE_MIN_EXPECTED_BPS = float(os.getenv("MIN_EXPECTED_ROI_BPS", "35.0"))
ROI_LOWVOL_BOOST_BPS      = float(os.getenv("ROI_LOWVOL_BOOST_BPS", "15.0"))
ROI_POORMETA_BOOST_BPS    = float(os.getenv("ROI_POORMETA_BOOST_BPS", "10.0"))
V_BPS_FLOOR               = float(os.getenv("V_BPS_FLOOR", "15.0"))

# Risk guards
MAX_DD_BPS_DAY    = float(os.getenv("MAX_DD_BPS_DAY", "-800.0"))
MAX_CONSEC_LOSS   = int(os.getenv("MAX_CONSEC_LOSS", "5"))
RISK_COOLDOWN_SEC = int(os.getenv("RISK_COOLDOWN_SEC", "900"))

CB_CONF_BUF: Dict[str, deque] = {s: deque(maxlen=max(50, CB_Q_WIN)) for s in SYMBOLS}
RISK_CONSEC_LOSS = 0
RISK_DAY_CUM_BPS = 0.0
RISK_LAST_RESET_DAY = time.strftime("%Y-%m-%d")


# =============================================================================
# =============== Binance USDT-M Futures (PUBLIC-ONLY) 클라이언트 ============
# =============================================================================
BINANCE_TESTNET = str(os.getenv("BINANCE_TESTNET","0")).lower() in ("1","true","y","yes")
class BinancePublicUM:
    def __init__(self, testnet: bool = False, timeout: int = 10):
        self.base = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        self.s = requests.Session()
        self.timeout = timeout
        self.s.headers.update({"User-Agent": "paper-binance-tcn-catboost/1.0"})

    def klines(self, symbol: str, interval: str = "1m", limit: int = 500):
        try:
            r = self.s.get(self.base + "/fapi/v1/klines",
                           params={"symbol": symbol, "interval": interval, "limit": int(limit)},
                           timeout=self.timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def book_ticker(self, symbol: str):
        try:
            r = self.s.get(self.base + "/fapi/v1/ticker/bookTicker",
                           params={"symbol": symbol}, timeout=self.timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def ticker_price(self, symbol: str):
        try:
            r = self.s.get(self.base + "/fapi/v1/ticker/price",
                           params={"symbol": symbol}, timeout=self.timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

# 인스턴스
mkt = BinancePublicUM(testnet=BINANCE_TESTNET, timeout=10)

# =============================================================================
# ========================= 스캐너/진단 출력 헬퍼 =============================
# =============================================================================
def _scan_tick(symbol: str):
    now = time.time()
    fields = []
    reasons = []

    has_pos = symbol in BROKER.positions()
    cd_left = max(0, int(COOLDOWN_UNTIL.get(symbol,0.0) - now))
    fields.append(f"pos={int(has_pos)}")
    fields.append(f"cool={cd_left}s")
    if has_pos: reasons.append("HAS_POS")
    if cd_left>0: reasons.append("COOLDOWN")

    # META
    if META_ON:
        hit, evr, mdd = _meta_stats(symbol)
        meta_ok = (hit >= META_MIN_HIT) and (evr >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
        meta_block = META_BLOCK_UNTIL.get(symbol,0.0) > now
        fields.append(f"meta_ok={int(meta_ok)} hit={hit:.2f} evr={evr:.1f} mdd={mdd:.1f} block={int(meta_block)}")
        if (not meta_ok) or meta_block:
            reasons.append("META")

    # TCN
    if TCN_MODEL is None:
        fields.append("tcn=OFF")
        reasons.append("TCN_OFF")
        t_side=None; mu=None; thr=_tcn_thr()
    else:
        try:
            t_side, mu, thr = tcn_direction(symbol)
            fields.append(f"tcn_side={t_side} mu={mu:.6g} thr={thr:.6g}")
            if t_side is None:
                reasons.append("NO_TCN_SIG")
        except Exception as e:
            fields.append(f"tcn_err={e}")
            reasons.append("TCN_ERR")
            t_side=None; mu=None; thr=_tcn_thr()

    cb_side, cb_conf = catboost_direction(symbol)
    fields.append(f"cb_side={cb_side} cb_conf={cb_conf:.2f}")
    if cb_side is None:
        reasons.append("CB_BLOCK")

    # 변동성 바닥
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    vol_ok = v_bps >= V_BPS_FLOOR
    fields.append(f"vol={v_bps:.1f}bps floor={V_BPS_FLOOR:.1f} {'OK' if vol_ok else 'LOW'}")
    if not vol_ok: reasons.append("LOW_VOL")

    # ROI 엣지(신호 있을 때만)
    if t_side in ("Buy","Sell"):
        bid, ask, mid = get_quote(symbol)
        if (t_side=="Buy" and ask>0) or (t_side=="Sell" and bid>0):
            entry_ref = float(ask if t_side=="Buy" else bid)
            tp1 = tp_from_bps(entry_ref, TP1_BPS, t_side)
            edge_bps = _bps_between(entry_ref, tp1)
            need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS
            ok = edge_bps >= need_bps
            fields.append(f"roi_edge={edge_bps:.1f} need={need_bps:.1f} {'OK' if ok else 'LOW'}")
            if not ok: reasons.append("LOW_ROI")

    # 연속 신호 버퍼
    dq = ENTRY_HISTORY[symbol]; need = max(1, N_CONSEC_SIGNALS)
    fields.append(f"consec={len(dq)}/{need} buf={list(dq)}")
    if need>1 and (len(dq)<need or any(x != (t_side or "") for x in dq)):
        reasons.append("CONSEC")

    verdict = "PASS" if not reasons else ("SKIP:" + ",".join(reasons))
    print(f"[SCAN] {symbol} | " + " | ".join(fields) + f" -> {verdict}")

def _whynot(symbol: str):
    msgs = []
    now = time.time()

    if COOLDOWN_UNTIL.get(symbol,0.0) > now:
        msgs.append(f"cooldown={int(COOLDOWN_UNTIL[symbol]-now)}s")
    msgs.append(f"has_position={int(symbol in BROKER.positions())}")

    if META_ON:
        hit, mean_bps, mdd = _meta_stats(symbol)
        meta_block = META_BLOCK_UNTIL.get(symbol,0.0) > now
        ok = (hit >= META_MIN_HIT) and (mean_bps >= META_MIN_EVR) and (mdd > META_MAX_DD_BPS)
        msgs.append(f"meta_ok={int(ok)} hit={hit:.2f} evr={mean_bps:.1f} mdd={mdd:.1f} block={int(meta_block)}")

    try:
        t_side, mu, thr = tcn_direction(symbol)
        if TCN_MODEL is None:
            msgs.append("tcn=OFF")
        else:
            msgs.append(f"tcn_side={t_side} mu={mu:.6g} thr={thr:.6g}")
    except Exception as e:
        msgs.append(f"tcn_err={e}")

    bid, ask, mid = get_quote(symbol)
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    msgs.append(f"vol={v_bps:.1f}bps floor={V_BPS_FLOOR:.1f} {'OK' if v_bps>=V_BPS_FLOOR else 'LOW'}")

    side_for_edge = t_side if 't_side' in locals() else None
    entry_ref = float(ask if side_for_edge=='Buy' else (bid if side_for_edge=='Sell' else mid))
    if side_for_edge in ("Buy","Sell") and entry_ref>0:
        tp1 = tp_from_bps(entry_ref, TP1_BPS, side_for_edge)
        edge_bps = _bps_between(entry_ref, tp1)
        need_bps = _entry_cost_bps() + _exit_cost_bps() + MIN_EXPECTED_ROI_BPS
        msgs.append(f"roi_edge={edge_bps:.1f} need={need_bps:.1f} {'OK' if edge_bps>=need_bps else 'LOW'}")

    dq = ENTRY_HISTORY[symbol]; need = max(1, N_CONSEC_SIGNALS)
    msgs.append(f"consec_need={need} buf={list(dq)}")

    print("[WHYNOT]", symbol, " | ".join(msgs))

# =============================================================================
# =========================== 시세/지표 헬퍼 ==================================
# =============================================================================
def get_quote(symbol: str) -> Tuple[float,float,float]:
    """
    Binance USDT-M:
      - /fapi/v1/ticker/bookTicker → bidPrice, askPrice
      - /fapi/v1/ticker/price      → price (last)
    """
    bt = mkt.book_ticker(symbol)
    bid = float((bt or {}).get("bidPrice") or 0.0)
    ask = float((bt or {}).get("askPrice") or 0.0)
    last = 0.0
    tp = mkt.ticker_price(symbol)
    if tp and tp.get("price") is not None:
        try: last = float(tp["price"])
        except Exception: last = 0.0
    mid = (bid+ask)/2.0 if (bid>0 and ask>0) else (last or bid or ask)
    return bid, ask, float(mid or 0.0)

def get_recent_1m(symbol: str, minutes: int = 200) -> Optional[pd.DataFrame]:
    rows = mkt.klines(symbol, interval=_bx_interval(INTERVAL), limit=min(int(minutes), 1000))
    if not rows: return None
    try:
        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(int(z[0]), unit="ms", utc=True),
            "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
            "close": float(z[4]),
            "volume": float(z[5]),
            "turnover": float(z[7]) if len(z) > 7 and z[7] is not None else float(z[5]) * float(z[4]),
        } for z in rows])
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None

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
TCN_CKPT = os.getenv("TCN_CKPT", "../multimodel/tcn_last.pt")
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
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, head_out=2):
        super().__init__()
        self.head_out = head_out
        layers=[]; ch_in=in_feat
        for i in range(levels):
            layers += [TemporalBlock(ch_in, hidden, k, 2**i, dropout)]
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, head_out)  # 1 or 2
    def forward(self, x):
        x = x.transpose(1,2)
        h = self.tcn(x)[:, :, -1]
        o = self.head(h)
        if self.head_out == 1:
            return None, o.squeeze(-1)
        else:
            return o[:,0], o[:,1]

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

TCN_HAS_MU = False
TCN_THR_PROB = float(os.getenv("TCN_THR_PROB","0.48"))

TCN_MODEL=None; TCN_CFG={}; TCN_MU=None; TCN_SD=None; TCN_SEQ_LEN=TCN_SEQ_LEN_FALLBACK
try:
    ck = torch.load(TCN_CKPT, map_location="cpu")
    state = ck.get("model", ck)
    TCN_CFG = ck.get("cfg", {})
    if "FEATS" in TCN_CFG: TCN_FEATS = TCN_CFG["FEATS"]
    TCN_SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))

    k0 = next(k for k in state.keys() if k.endswith("tcn.0.conv1.weight_v") or k.endswith("tcn.0.conv1.weight"))
    w0 = state[k0]
    hidden = w0.shape[0]
    in_feat = w0.shape[1]
    k_size = w0.shape[2]
    levels = 1 + max(int(k.split('.')[1]) for k in state.keys() if k.startswith("tcn.") and ".conv1." in k)
    head_w = state.get("head.weight")
    head_out = head_w.shape[0] if head_w is not None else 2

    TCN_MODEL = TCNBinary(in_feat=in_feat, hidden=hidden, levels=levels, k=k_size,
                          dropout=0.1, head_out=head_out).eval()
    TCN_MODEL.load_state_dict(state, strict=True)
    try:
        TCN_HEAD = "cls" if TCN_MODEL.head.out_features == 1 else "reg"
    except Exception:
        TCN_HEAD = "reg"
    TCN_MU = ck.get("scaler_mu", None)
    TCN_SD = ck.get("scaler_sd", None)
except Exception as e:
    print(f"[WARN] TCN load failed: {e}")
    TCN_MODEL=None

def _tcn_thr():
    mode = os.getenv("THR_MODE", "fixed").lower()
    if mode == "fee" and TCN_HAS_MU:
        return float(fee_threshold())
    return float(os.getenv("MODEL_THR_BPS", "5.0")) / 1e4

@torch.no_grad()
def tcn_direction(symbol: str):
    if TCN_MODEL is None:
        return None, 0.0, _tcn_thr()

    df = get_recent_1m(symbol, minutes=TCN_SEQ_LEN + 10)
    if df is None or len(df) < (TCN_SEQ_LEN+1):
        return None, 0.0, _tcn_thr()

    feats = _build_tcn_features(df)
    X = feats[TCN_FEATS].tail(TCN_SEQ_LEN).to_numpy().astype(np.float32)
    X = _apply_tcn_scaler(X, TCN_MU, TCN_SD)
    x_t = torch.from_numpy(X[None, ...])

    y_hat, logit = TCN_MODEL(x_t)

    if y_hat is None:
        p = float(torch.sigmoid(logit).item())
        prob_thr = float(os.getenv("LOGIT_PROB_THR", "0.42"))
        side = "Buy" if p > prob_thr else ("Sell" if p < (1.0 - prob_thr) else None)
        score = p - 0.5
        thr = prob_thr - 0.5
        if VERBOSE: print(f"[DEBUG][TCN][{symbol}] p={p:.3f} thr={prob_thr:.2f} -> {side}")
        return side, score, thr

    mu = float(y_hat.item())
    thr = _tcn_thr()
    side = "Buy" if mu > thr else ("Sell" if mu < -thr else None)
    if VERBOSE: print(f"[DEBUG][TCN][{symbol}] mu={mu:.6g} thr={thr:.6g} -> {side}")
    return side, mu, thr

@torch.no_grad()
def tcn_time_to_target_minutes(symbol: str, side: str, target_bps: float):
    if TCN_HEAD == "cls":
        return None
    t_side, mu_t, _thr = tcn_direction(symbol)
    if mu_t is None: return None
    step = abs(float(mu_t))
    if step <= 1e-8: return None
    thr = _log_thr_from_bps(target_bps)
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

# ==== PATCH: helper for dynamic CB threshold ====
def _cb_dynamic_cut(symbol: str, fallback_thr: float = MIN_CB_THR) -> float:
    """Return a dynamic probability cut from recent CB confidences (quantile)."""
    buf = CB_CONF_BUF.setdefault(symbol, deque(maxlen=max(50, CB_Q_WIN)))
    if not buf:
        return float(fallback_thr)
    arr = sorted(buf)
    k = int(min(max(0, round((len(arr)-1) * CB_Q_PERC)), len(arr)-1))
    q = float(arr[k])
    return max(float(fallback_thr), q)


# ==== PATCH: catboost_direction (drop-in replacement) ====
@torch.no_grad()
def catboost_direction(symbol: str) -> Tuple[Optional[str], float]:
    """
    TCN+CatBoost 합의(진입 판단) + 동적 임계치(최근 분위수) 적용
    리턴: (side|None, confidence)
    """
    def _log(msg: str):
        if VERBOSE or TRACE:
            print(msg)

    if CB_MODEL is None or not CB_META:
        _log(f"[CB][{symbol}] SKIP: NO_CB_MODEL_OR_META")
        return (None, 0.0)

    minutes = max(300, int(TCN_SEQ_LEN + 80))
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < (TCN_SEQ_LEN + 61):
        _log(f"[CB][{symbol}] SKIP: SHORT_DF len={0 if df is None else len(df)} need>={TCN_SEQ_LEN+61}")
        return (None, 0.0)

    # --- TCN gate ---
    try:
        t_side, t_score, t_thr = tcn_direction(symbol)  # score: μ or (p-0.5)
    except Exception as e:
        _log(f"[CB][{symbol}] SKIP: TCN_ERR {e}")
        return (None, 0.0)
    t_ok = (t_side in ("Buy","Sell")) and (abs(float(t_score)) >= float(t_thr))
    _log(f"[CB][{symbol}][TCN] side={t_side} score={t_score:.6g} thr={t_thr:.6g} ok={int(t_ok)}")

    # --- Base feats ---
    base = _latest_base_feats(df)
    if base is None:
        _log(f"[CB][{symbol}] SKIP: BASE_FEATS_NA")
        return (None, 0.0)

    feat_cols: List[str] = CB_META.get("feature_cols", [])
    cat_cols:  List[str] = CB_META.get("cat_cols", ["symbol"])
    base_thr:  float      = float(os.getenv("CB_THR", CB_META.get("threshold_prob", 0.55)))
    # optional TCN-derived features
    row = dict(base)
    if "tcn_mu" in feat_cols:
        row["tcn_mu"] = float(t_score)
    if "tcn_logit" in feat_cols and "tcn_logit" not in row:
        row["tcn_logit"] = 0.0

    missing = [c for c in feat_cols if c not in row]
    if missing:
        _log(f"[CB][{symbol}] SKIP: MISSING_FEATS {missing}")
        return (None, 0.0)

    X = pd.DataFrame([[row[c] for c in feat_cols] + [symbol]*len(cat_cols)],
                     columns=feat_cols + cat_cols)
    pool = Pool(X, cat_features=[len(feat_cols)+i for i in range(len(cat_cols))])

    try:
        proba = CB_MODEL.predict_proba(pool)[0]
    except Exception as e:
        _log(f"[CB][{symbol}] SKIP: CB_ERR {e}")
        return (None, 0.0)

    pred_idx = int(np.argmax(proba))
    inv_map = {0:-1, 1:0, 2:+1}
    y_hat = inv_map.get(pred_idx, 0)
    cb_conf = float(np.max(proba))
    cb_side = "Buy" if y_hat > 0 else ("Sell" if y_hat < 0 else None)

    # push to buffer BEFORE deciding to keep distribution fresh
    CB_CONF_BUF.setdefault(symbol, deque(maxlen=max(50, CB_Q_WIN))).append(cb_conf)

    # dynamic threshold: max(base, recent-quantile)
    dyn_thr = _cb_dynamic_cut(symbol, fallback_thr=base_thr)
    thr_prob = max(float(MIN_CB_THR), float(dyn_thr))
    _log(f"[CB][{symbol}][CB] pred={cb_side or 'NoTrade'} conf={cb_conf:.3f} thr(base={base_thr:.2f}, dyn={dyn_thr:.2f}) -> use={thr_prob:.2f}")

    reasons = []
    if not t_ok:
        reasons.append("TCN_FAIL")
    if cb_side is None:
        reasons.append("CB_NOTRADE")
    if cb_conf < thr_prob:
        reasons.append("CB_PROB_LOW")
    if t_ok and cb_side not in (None,) and cb_side != t_side:
        reasons.append("DISAGREE")

    if not reasons:
        final_side = cb_side
        if ENTRY_MODE == "inverse":
            final_side = "Sell" if cb_side == "Buy" else "Buy"
        _log(f"[CB][{symbol}] -> PASS {final_side} (conf={cb_conf:.3f})")
        return (final_side, cb_conf)

    _log(f"[CB][{symbol}] -> SKIP: {','.join(reasons)}")
    return (None, cb_conf)

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

# ==== PATCH: _entry_allowed_by_filters (drop-in replacement) ====
def _entry_allowed_by_filters(symbol: str, side: str, entry_ref: float) -> bool:
    # 1) volatility floor
    v_bps = recent_vol_bps(symbol, VOL_WIN)
    if v_bps < V_BPS_FLOOR:
        if VERBOSE: print(f"[SKIP] {symbol} vol {v_bps:.1f}bps < floor {V_BPS_FLOOR}bps")
        return False

    # 2) base ROI requirement (fees + slippage + base edge)
    base_need = _entry_cost_bps() + _exit_cost_bps() + ROI_BASE_MIN_EXPECTED_BPS

    # 3) dynamic boosts
    need_bps = base_need
    # low-vol boost: if 변동성 낮으면 더 높은 엣지 요구
    if v_bps < (2.0 * V_BPS_FLOOR):
        need_bps += ROI_LOWVOL_BOOST_BPS
    # meta deterioration boost
    if META_ON:
        hit, evr, mdd = _meta_stats(symbol)
        if (hit < META_MIN_HIT) or (evr < META_MIN_EVR) or (mdd <= META_MAX_DD_BPS):
            need_bps += ROI_POORMETA_BOOST_BPS

    # 4) compute edge to TP1
    tp1_px = tp_from_bps(entry_ref, TP1_BPS, side)
    edge_bps = _bps_between(entry_ref, tp1_px)

    if edge_bps < need_bps:
        if VERBOSE: print(f"[SKIP] {symbol} ROI edge {edge_bps:.1f} < need {need_bps:.1f}bps (v={v_bps:.1f})")
        return False
    return True
# ==== PATCH: risk guard on close ====
def _risk_guard_on_close(symbol: str, pnl_bps: float):
    """Update day/consecutive loss guards and enforce cooldown if breached."""
    global RISK_CONSEC_LOSS, RISK_DAY_CUM_BPS, RISK_LAST_RESET_DAY, COOLDOWN_UNTIL
    today = time.strftime("%Y-%m-%d")
    if today != RISK_LAST_RESET_DAY:
        RISK_LAST_RESET_DAY = today
        RISK_DAY_CUM_BPS = 0.0
        RISK_CONSEC_LOSS = 0

    RISK_DAY_CUM_BPS += float(pnl_bps)
    if pnl_bps <= 0:
        RISK_CONSEC_LOSS += 1
    else:
        RISK_CONSEC_LOSS = 0

    breach = (RISK_DAY_CUM_BPS <= MAX_DD_BPS_DAY) or (RISK_CONSEC_LOSS >= MAX_CONSEC_LOSS)
    if breach:
        until = time.time() + RISK_COOLDOWN_SEC
        for s in SYMBOLS:
            COOLDOWN_UNTIL[s] = until
        if VERBOSE:
            print(f"[RISK] cooldown {RISK_COOLDOWN_SEC}s "
                  f"(day_dd={RISK_DAY_CUM_BPS:.1f}bps, consec={RISK_CONSEC_LOSS})")


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
    if TRACE:
        _scan_tick(symbol)
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

# ==== PATCH: close_and_log (drop-in replacement; calls _risk_guard_on_close) ====
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
        print(f"[{reason}] {symbol} pnl_bps={pnl_bps:.1f} gross={gross:.6f} fee={fee:.6f} eq={new_eq:.2f}")
    POSITION_DEADLINE.pop(symbol, None)
    STATE.pop(symbol, None)
    COOLDOWN_UNTIL[symbol] = time.time() + COOLDOWN_SEC
    _risk_guard_on_close(symbol, pnl_bps)

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
    print(f"[START] PAPER TCN+CatBoost | Binance USDT-M | TESTNET={BINANCE_TESTNET}")
    _log_equity(get_wallet_equity())
    _log_equity_mtm(get_equity_mtm())

    if DIAG:
        for s in SYMBOLS:
            _whynot(s)
        return

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
