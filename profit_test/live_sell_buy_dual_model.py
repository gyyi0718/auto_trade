# -*- coding: utf-8 -*-
# PAPER 전용 (신호 전용 모드): DeepAR + TCN 듀얼 합의
# - 주문/포지션/CSV 기록 없음
# - 3초(기본)마다 실시간 호가 조회 + Buy/Sell 신호만 출력

import os, time, math, csv, random, threading, collections
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from pybit.unified_trading import HTTP
from concurrent.futures import ThreadPoolExecutor
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
import warnings, logging
warnings.filterwarnings("ignore", module="lightning")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ===== ENV =====
CATEGORY = "linear"
INTERVAL = "1"  # 1m

SYMBOLS = os.getenv("SYMBOLS","ASTERUSDT").split(",")
ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model|inverse|random
MODEL_MODE = os.getenv("MODEL_MODE", "online").lower()   # fixed | online
ONLINE_LR  = float(os.getenv("ONLINE_LR", "0.05"))
ONLINE_DECAY = float(os.getenv("ONLINE_DECAY", "0.97"))
ONLINE_MAXBUF = int(os.getenv("ONLINE_MAXBUF", "240"))

LEVERAGE = float(os.getenv("LEVERAGE","5"))  # 신호엔 직접 영향 없음
TAKER_FEE = float(os.getenv("TAKER_FEE","0.0006"))
SLIPPAGE_BPS_TAKER = float(os.getenv("SLIPPAGE_BPS_TAKER","1.0"))
FEE_SAFETY = float(os.getenv("FEE_SAFETY","1.2"))

TP1_BPS = float(os.getenv("TP1_BPS","50.0"))  # ROI 임계치 계산용
V_BPS_FLOOR = float(os.getenv("V_BPS_FLOOR","5.0"))

VOL_WIN = int(os.getenv("VOL_WIN", "60"))
PRED_HORIZ_MIN = int(os.getenv("PRED_HORIZ_MIN","1"))

VERBOSE = str(os.getenv("VERBOSE","1")).lower() in ("1","y","yes","true")
POLL_SEC = float(os.getenv("POLL_SEC","3"))

# ===== MARKET (public only) =====
TESTNET = str(os.getenv("BYBIT_TESTNET","0")).lower() in ("1","true","y","yes")
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

# === Online Adapter ===
class OnlineAdapter:
    def __init__(self, lr=0.05, decay=0.97, maxbuf=240):
        self.a, self.b = 1.0, 0.0
        self.lr = lr; self.decay = decay
        self.buf = collections.deque(maxlen=maxbuf)
    def update(self, mu, y):
        pred = self.a*mu + self.b
        e = (pred - y)
        self.a -= self.lr * e * mu
        self.b -= self.lr * e
        self.a = 0.999*self.a + 0.001*1.0
        self.b = 0.999*self.b
        self.buf.append((float(mu), float(y)))
    def transform(self, mu):
        return self.a*mu + self.b

_ADAPTERS: Dict[str, OnlineAdapter] = collections.defaultdict(
    lambda: OnlineAdapter(ONLINE_LR, ONLINE_DECAY, ONLINE_MAXBUF)
)

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
        "series_id": symbol,
        "open": float(z[1]), "high": float(z[2]), "low": float(z[3]),
        "close": float(z[4]), "volume": float(z[5]), "turnover": float(z[6]),
    } for z in rows])
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    return df

def recent_vol_bps(symbol: str, minutes: int = None) -> float:
    minutes = minutes or VOL_WIN
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 5:
        return 10.0
    v_bps = float(df["log_return"].abs().median() * 1e4)
    return max(v_bps, 1.0)

def fee_threshold(taker_fee=TAKER_FEE, slip_bps=SLIPPAGE_BPS_TAKER, safety=FEE_SAFETY):
    rt = 2.0*taker_fee + 2.0*(slip_bps/1e4)
    return math.log(1.0 + safety*rt)

def _bps_between(p1:float,p2:float)->float:
    return abs((p2 - p1) / max(p1, 1e-12)) * 10_000.0

# ===== DeepAR =====
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer
import torch.nn.functional as F
from pytorch_forecasting.metrics import NormalDistributionLoss

SEQ_LEN = int(os.getenv("SEQ_LEN","240"))
PRED_LEN = int(os.getenv("PRED_LEN","60"))
MODEL_CKPT = os.getenv("MODEL_CKPT","../multimodel/models/multi_deepar_best.ckpt")

class SignAwareNormalLoss(NormalDistributionLoss):
    def __init__(self, reduction="mean", lambda_sign=0.3, alpha_sign=7.0):
        super().__init__(reduction=reduction); self.lambda_sign=float(lambda_sign); self.alpha_sign=float(alpha_sign)
    def _unpack_target(self, target):
        if isinstance(target,(tuple,list)): y=target[0]; w=target[1] if len(target)>1 else None
        elif isinstance(target,dict): y=target.get("target",target.get("y",target)); w=target.get("weight",None)
        else: y=target; w=None
        return y,w
    def forward(self,y_pred,target):
        base=super().forward(y_pred,target); y,w=self._unpack_target(target); mu=self.to_prediction(y_pred)
        if mu.shape!=y.shape: y=y.view_as(mu)
        logits=self.alpha_sign*mu; y_pos=(y>0).float()
        bce=F.binary_cross_entropy_with_logits(logits,y_pos,reduction="none")
        if w is not None:
            if w.shape!=bce.shape: w=w.view_as(bce)
            denom=torch.clamp(w.sum(),min=1.0); bce=(bce*w).sum()/denom
        else: bce=bce.mean()
        return base + self.lambda_sign*bce

DEEPar = None
try:
    DEEPar = DeepAR.load_from_checkpoint(MODEL_CKPT, map_location="cpu").eval()
except Exception as e:
    if VERBOSE: print(f"[WARN] DeepAR load failed: {e}")

def ma_side(symbol: str, minutes: int = 180) -> Optional[str]:
    df = get_recent_1m(symbol, minutes=minutes)
    if df is None or len(df) < 60:
        return None
    c = df["close"].to_numpy()
    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:])
    if ma20 > ma60: return "Buy"
    if ma20 < ma60: return "Sell"
    return None

def build_infer_dataset(df_sym: pd.DataFrame):
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

@torch.no_grad()
def deepar_raw(symbol: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """DeepAR 방향과 임계치 반환. mu는 1-스텝 평균 로그수익 예측치."""
    base = None; mu_adj=None; thr=None
    try:
        if DEEPar is not None:
            df = get_recent_1m(symbol, minutes=SEQ_LEN+PRED_LEN+10)
            if df is not None and len(df) >= (SEQ_LEN+1):
                tsd = build_infer_dataset(df)
                dl = tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
                mu = float(DEEPar.predict(dl, mode="prediction")[0, 0])
                mu_adj = _ADAPTERS[symbol].transform(mu) if MODEL_MODE=="online" else mu
                thr = fee_threshold()
                if VERBOSE: print(f"[DEBUG][DEEPar][{symbol}] mu={mu:.6g} mu'={mu_adj:.6g} thr={thr:.6g}")
                if   mu_adj >  thr: base = "Buy"
                elif mu_adj < -thr: base = "Sell"
    except Exception as e:
        if VERBOSE: print(f"[PRED][ERR] {symbol} -> {e}")
    if base is None:
        base = ma_side(symbol)
    return base, mu_adj, thr

@torch.no_grad()
def deepar_direction(symbol: str):
    base, _, _ = deepar_raw(symbol)
    if ENTRY_MODE == "model":
        side = base
    elif ENTRY_MODE == "inverse":
        if   base == "Buy": side = "Sell"
        elif base == "Sell": side = "Buy"
        else: side = None
    elif ENTRY_MODE == "random":
        side = base or random.choice(["Buy","Sell"])
    else:
        side = base
    conf = 1.0 if side is not None else 0.0
    return side, conf

# ===== TCN (binary-threshold driver) =====
import torch.nn as nn

TCN_CKPT = os.getenv("TCN_CKPT", "../multimodel/models/tcn_best.pt")
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
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1):
        super().__init__()
        layers=[]; ch_in=in_feat
        for i in range(levels):
            layers += [TemporalBlock(ch_in, hidden, k, 2**i, dropout)]
            ch_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # [reg_mu, logit]
    def forward(self, x):
        x = x.transpose(1,2)
        h = self.tcn(x)[:, :, -1]
        o = self.head(h)
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

TCN_MODEL = None; TCN_CFG={}; TCN_MU=None; TCN_SD=None
try:
    _ckpt = torch.load(TCN_CKPT, map_location="cpu")
    TCN_CFG = _ckpt.get("cfg", {})
    feats = TCN_CFG.get("FEATS", TCN_FEATS)
    in_feat = len(feats)
    TCN_MODEL = TCNBinary(in_feat=in_feat, hidden=128, levels=6, k=3, dropout=0.1).eval()
    TCN_MODEL.load_state_dict(_ckpt["model"], strict=False)
    TCN_MU = _ckpt.get("scaler_mu", None)
    TCN_SD = _ckpt.get("scaler_sd", None)
    TCN_FEATS = feats
    TCN_SEQ_LEN = int(TCN_CFG.get("SEQ_LEN", TCN_SEQ_LEN_FALLBACK))
except Exception as e:
    TCN_MODEL = None
    TCN_SEQ_LEN = TCN_SEQ_LEN_FALLBACK
    print(f"[WARN] TCN load failed: {e}")

def _tcn_thr():
    mode = os.getenv("THR_MODE", "fixed").lower()
    if mode == "fee":
        return float(fee_threshold())
    bps = float(os.getenv("MODEL_THR_BPS", "5.0"))
    return bps / 1e4

@torch.no_grad()
def tcn_direction(symbol: str) -> Tuple[Optional[str], float, float]:
    if TCN_MODEL is None:
        return (None, 0.0, _tcn_thr())
    df = get_recent_1m(symbol, minutes=TCN_SEQ_LEN + 10)
    if df is None or len(df) < (TCN_SEQ_LEN+1):
        return (None, 0.0, _tcn_thr())
    feats = _build_tcn_features(df)
    X = feats[TCN_FEATS].tail(TCN_SEQ_LEN).to_numpy().astype(np.float32)
    X = _apply_tcn_scaler(X, TCN_MU, TCN_SD)
    x_t = torch.from_numpy(X[None, ...])
    mu, _ = TCN_MODEL(x_t)
    mu = float(mu.item())
    thr = _tcn_thr()
    if   mu >  thr: side = "Buy"
    elif mu < -thr: side = "Sell"
    else:           side = None
    if VERBOSE: print(f"[DEBUG][TCN][{symbol}] mu={mu:.6g} thr={thr:.6g} -> {side}")
    return side, mu, thr

@torch.no_grad()
def dual_direction_deepar_tcn(symbol: str) -> Tuple[Optional[str], float, dict]:
    d_side, _ = deepar_direction(symbol)
    t_side, mu_t, thr_t = tcn_direction(symbol)
    side = None
    if d_side in ("Buy","Sell") and t_side == d_side and abs(mu_t) >= thr_t:
        side = d_side
    # ENTRY_MODE 반영
    m = (ENTRY_MODE or "model").lower()
    if side is not None:
        if m == "inverse":
            side = "Sell" if side=="Buy" else "Buy"
        elif m == "random":
            side = side
    debug = {"deepar": d_side, "tcn": t_side, "tcn_mu": mu_t, "tcn_thr": thr_t}
    return side, (1.0 if side else 0.0), debug

def choose_entry(symbol: str):
    side, conf, debug = dual_direction_deepar_tcn(symbol)
    if side in ("Buy","Sell"):
        return side, conf, debug
    # 듀얼 미충족 시 DeepAR 단독
    d_side, d_conf = deepar_direction(symbol)
    return d_side, d_conf, {"deepar": d_side, "tcn": None, "tcn_mu": None, "tcn_thr": _tcn_thr()}

# ===== Signal-only loop =====
def signal_loop(poll_sec: float = 3.0):
    print(f"[START] SIGNAL ONLY | MODE={ENTRY_MODE} TESTNET={TESTNET} POLL={poll_sec}s")
    while True:
        ts = int(time.time())
        for s in SYMBOLS:
            bid, ask, mid = get_quote(s)
            side, conf, dbg = choose_entry(s)
            # 저변동 필터 참고용 메시지 (선택)
            v_bps = recent_vol_bps(s, VOL_WIN)
            if v_bps < V_BPS_FLOOR and VERBOSE:
                print(f"[VOL] {s} {v_bps:.1f}bps < floor {V_BPS_FLOOR:.1f}bps")

            sig = side if side in ("Buy","Sell") else "None"
            print(f"[{ts}] {s} SIGNAL={sig} mid={mid:.6f} bid={bid:.6f} ask={ask:.6f} ->"
                  f"(deepar={dbg.get('deepar')} tcn={dbg.get('tcn')} mu={dbg.get('tcn_mu')} thr={dbg.get('tcn_thr')})")
        time.sleep(poll_sec)

# ===== Main =====
def main():
    signal_loop(POLL_SEC)

if __name__ == "__main__":
    main()
