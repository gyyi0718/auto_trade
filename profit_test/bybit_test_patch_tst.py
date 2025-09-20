# bybit_test_patchtst.py
# -*- coding: utf-8 -*-
"""
Bybit 페이퍼트레이딩 + PatchTST 신호
- 체크포인트 호환 로더: head_cls 구조 자동 감지(Sequential ↔ Attn+Max)
- 환경변수 PATCHTST_CKPT 로 ckpt 경로 지정 가능. 없으면 ./ckpt/patchtst_best_ep*.pt 검색.
"""
import os, time, math, csv, threading, requests, signal, glob
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
SYMBOLS = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "LINKUSDT","ADAUSDT","SUIUSDT","1000PEPEUSDT","MNTUSDT",
]
LEVERAGE_LIST   = [10, 25, 100]
START_CASH_LIST = [100.0, 500.0, 1000.0]

ENTRY_MODE = os.getenv("ENTRY_MODE", "model")  # model | inverse | random
MAX_OPEN   = 6
RISK_PCT_OF_EQUITY = 0.30
ENTRY_PORTION      = 0.40
MIN_NOTIONAL_USDT  = 10.0
MAX_NOTIONAL_ABS   = 2000.0
MAX_NOTIONAL_PCT_OF_EQUITY = 20.0

TAKER_FEE    = 0.0006
SLIP_BPS_IN  = 1.0
SLIP_BPS_OUT = 1.0

TP1_BPS, TP2_BPS, TP3_BPS = 50.0, 100.0, 200.0
TP1_RATIO, TP2_RATIO = 0.40, 0.35
BE_AFTER_TIER = 2
BE_EPS_BPS    = 2.0
TRAIL_BPS     = 50.0
TRAIL_AFTER_TIER = 2
SL_ROI_PCT    = 0.01
MAX_HOLD_SEC  = 3600
COOLDOWN_SEC  = 10

TICK_INTERVAL_SEC   = 1.0
USE_MARK_PRICE_FOR_RISK = True
MAX_MARK_FAIR_DIFF  = 0.01

VERBOSE              = True
HEARTBEAT_SEC        = 5.0
WORKER_STATUS_EVERY  = 5

OUT_DIR = "./logs_sync"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Bybit v5 Public REST
# =========================
BYBIT_API = "https://api.bybit.com"

def _get(path: str, params: dict=None, retries: int=3):
    url = BYBIT_API + path
    params = params or {}
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and j.get("retCode") == 0:
                    return j.get("result")
                last_err = f"retCode={j.get('retCode')} retMsg={j.get('retMsg')}"
            else:
                last_err = f"HTTP {r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.1)
    raise RuntimeError(f"HTTP GET failed: {path} {params} -> {last_err}")

def _iv_map_1m(iv: str) -> str:
    iv = str(iv).lower().strip()
    return {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
            "1h":"60","2h":"120","4h":"240","1d":"D"}.get(iv, "1")

def get_all_book_tickers() -> Dict[str, Tuple[float,float,float,float]]:
    res = _get("/v5/market/tickers", {"category":"linear"})
    items = (res or {}).get("list", []) if isinstance(res, dict) else []
    out = {}
    u = set(SYMBOLS)
    for it in items:
        sym = it.get("symbol")
        if sym not in u: continue
        bid = float(it.get("bid1Price") or 0)
        ask = float(it.get("ask1Price") or 0)
        last = float(it.get("lastPrice") or 0)
        mark = float(it.get("markPrice") or 0)
        if bid>0 and ask>0:
            mid = (bid+ask)/2.0
        elif last>0:
            mid = last
            if bid==0: bid = last
            if ask==0: ask = last
        else:
            continue
        out[sym] = (bid, ask, mid, mark)
    return out

def get_mark_price(symbol: str) -> float:
    res = _get("/v5/market/tickers", {"category":"linear", "symbol":symbol})
    items = (res or {}).get("list", [])
    if not items: return 0.0
    return float(items[0].get("markPrice") or 0.0)

def get_klines(symbol: str, interval: str="1m", limit: int=120) -> pd.DataFrame:
    iv = _iv_map_1m(interval)
    res = _get("/v5/market/kline", {"category":"linear","symbol":symbol,"interval":iv,"limit":int(min(1500,limit))})
    rows = (res or {}).get("list", [])
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["start"].astype(np.int64), unit="ms", utc=True)
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    return df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)

# =========================
# Utils
# =========================
def _opp(side: str) -> str: return "Sell" if side=="Buy" else "Buy"
def _roi_pct(side: str, entry: float, px: float) -> float:
    return (px/entry - 1.0) if side=="Buy" else (entry/px - 1.0)
def _round_down_qty(q: float, step: float) -> float:
    if step<=0: return q
    return math.floor(q/step)*step

SYMBOL_RULES = {
    "BTCUSDT": {"tick":0.5,   "step":0.001, "min":0.001},
    "ETHUSDT": {"tick":0.05,  "step":0.001, "min":0.001},
    "SOLUSDT": {"tick":0.001, "step":0.01,  "min":0.01},
    "XRPUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
    "DOGEUSDT":{"tick":0.00001,"step":1.0,  "min":1.0},
    "LINKUSDT":{"tick":0.0005,"step":0.01,  "min":0.01},
    "ADAUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
    "SUIUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
    "1000PEPEUSDT":{"tick":0.00001,"step":1.0,"min":1.0},
    "MNTUSDT": {"tick":0.0001,"step":1.0,   "min":1.0},
}
DEFAULT_RULE = {"tick":0.0001,"step":0.001,"min":0.001}
def rule_of(sym:str): return SYMBOL_RULES.get(sym, DEFAULT_RULE)

# =========================
# PatchTST (호환 헤드)
# =========================
import torch
import torch.nn as nn

FEATURE_COLS = ["f_ret1","f_ret5","f_ret15","f_rv5","f_rv15","f_v5","f_v15","f_spr"]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):  # [B,L,d]
        return x + self.pe[:, :x.size(1), :]

class ClassifierHeadAttnMax(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        self.fc = nn.Sequential(
            nn.Linear(d*2, d//2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d//2, 1)
        )
    def forward(self, z_tokens):   # [B,T,d]
        w = torch.softmax(self.attn(z_tokens).squeeze(-1), dim=1)
        attn_pool = (w.unsqueeze(-1) * z_tokens).sum(1)
        max_pool  = z_tokens.max(dim=1).values
        h = torch.cat([attn_pool, max_pool], dim=-1)
        return self.fc(h)

class ClassifierHeadSequential(nn.Module):
    """옛날 체크포인트(head_cls.0/3.*)와 동일 구조"""
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.seq = nn.Sequential(
            nn.Linear(d, d//2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d//2, 1)
        )
    def forward(self, z_tokens):   # [B,T,d]
        z = self.pool(z_tokens.transpose(1,2)).squeeze(-1)  # [B,d]
        return self.seq(z)

class PatchTST(nn.Module):
    def __init__(self, n_features:int, cfg:dict, head_type:str="auto"):
        super().__init__()
        self.cfg = cfg
        P, S = int(cfg["patch_len"]), int(cfg["patch_stride"])
        self.input_len = int(cfg["input_len"])
        self.n_patches = 1 + (self.input_len - P)//S
        self.patch_dim = P * n_features
        d_model = int(cfg["d_model"]); n_heads = int(cfg["n_heads"]); n_layers = int(cfg["n_layers"])
        drop = float(cfg.get("dropout", 0.1))
        self.embed = nn.Linear(self.patch_dim, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=self.n_patches)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=d_model*4, dropout=drop,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool_mean = nn.AdaptiveAvgPool1d(1)
        self.head_q = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d_model, len(cfg["horizons"])*len(cfg["quantiles"]))
        )
        # 분류 헤드는 ckpt 로드시 결정
        self._head_type = head_type
        self.head_cls = None

    def _ensure_head(self, head_type: str, d_model: int, drop: float):
        if head_type == "attnmax":
            self.head_cls = ClassifierHeadAttnMax(d_model, drop=drop)
        else:
            self.head_cls = ClassifierHeadSequential(d_model, drop=drop)

    def _patchify(self, x):  # x:[B,L,F] -> [B,T,P*F]
        B,L,F = x.shape; P,S = int(self.cfg["patch_len"]), int(self.cfg["patch_stride"])
        T = 1 + (L - P)//S
        xs=[]
        for t in range(T):
            s=t*S
            xs.append(x[:, s:s+P, :].reshape(B, -1))
        return torch.stack(xs, dim=1)

    def forward(self, x):
        patches = self._patchify(x)
        z = self.embed(patches)
        z = self.posenc(z)
        z = self.encoder(z)
        z_mean = self.pool_mean(z.transpose(1,2)).squeeze(-1)
        q = self.head_q(z_mean)
        c = self.head_cls(z)
        return q, c

# =========================
# 특징 생성
# =========================
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    p = g["close"].astype(float).replace(0, np.nan).ffill()
    logp = np.log(p)
    r1 = logp.diff(); r5 = logp.diff(5); r15 = logp.diff(15)
    rv5 = (r1**2).rolling(5).sum(); rv15 = (r1**2).rolling(15).sum()
    vol = g["volume"].astype(float).fillna(0.0)
    v5 = vol.rolling(5).mean(); v15 = vol.rolling(15).mean()
    spr = (g["high"] - g["low"]).astype(float) / g["close"].replace(0,np.nan)
    spr = spr.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    g["f_ret1"]=r1.fillna(0.0); g["f_ret5"]=r5.fillna(0.0); g["f_ret15"]=r15.fillna(0.0)
    g["f_rv5"]=rv5.fillna(0.0); g["f_rv15"]=rv15.fillna(0.0)
    g["f_v5"]=v5.fillna(0.0);   g["f_v15"]=v15.fillna(0.0)
    g["f_spr"]=spr.fillna(0.0)
    return g

def _find_ckpt(path_or_glob: str) -> Optional[str]:
    if os.path.isfile(path_or_glob): return path_or_glob
    files = sorted(glob.glob(path_or_glob))
    return files[-1] if files else None

# =========================
# Runner
# =========================
class PatchTSTRunner:
    def __init__(self, ckpt_path: Optional[str]=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision("high")
        if ckpt_path is None:
            ckpt_path = os.getenv("PATCHTST_CKPT") or _find_ckpt("./ckpt/patchtst_best_ep*.pt")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError("PatchTST checkpoint not found. Set env PATCHTST_CKPT or place under ./ckpt/")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cfg = ckpt.get("cfg") or {}
        self.horizons  = tuple(self.cfg.get("horizons",(1,5,15,60)))
        self.quantiles = tuple(self.cfg.get("quantiles",(0.1,0.5,0.9)))
        self.input_len = int(self.cfg.get("input_len",240))
        self.tp_bps = int(self.cfg.get("tp_bps",120))
        self.sl_bps = int(self.cfg.get("sl_bps",90))

        sd = ckpt["model"]
        keys = set(sd.keys())
        # 헤드 타입 자동 판별
        if any(k.startswith("head_cls.attn.") or k.startswith("head_cls.fc.") for k in keys):
            head_type = "attnmax"
        elif any(k.startswith("head_cls.0.") or k.startswith("head_cls.3.") for k in keys):
            head_type = "seq"
        else:
            head_type = "seq"

        d_model = int(self.cfg.get("d_model",192))
        drop = float(self.cfg.get("dropout",0.1))
        self.model = PatchTST(n_features=len(FEATURE_COLS), cfg=self.cfg, head_type=head_type).to(self.device)
        self.model._ensure_head(head_type, d_model, drop)

        # 로드 시 strict=True로 시도, 실패하면 strict=False로 마지막 시도
        try:
            self.model.load_state_dict(sd, strict=True)
        except Exception as e1:
            if VERBOSE:
                print(f"[PatchTST][INIT][WARN] strict load failed ({head_type}): {e1}")
            self.model.load_state_dict(sd, strict=False)

        self.model.eval()
        # 의사결정 파라미터
        self.cost_bps  = float(os.getenv("PATCHTST_COST_BPS", "10"))
        self.theta_bps = float(os.getenv("PATCHTST_THETA_BPS", "20"))
        self.band_max  = float(os.getenv("PATCHTST_BAND_MAX_BPS", "60"))
        self.use_h_idx = int(os.getenv("PATCHTST_H_IDX", "1"))  # 0:1m,1:5m,2:15m,3:60m

    @torch.no_grad()
    def predict_symbol(self, symbol: str) -> Tuple[Optional[str], float, dict]:
        need = max(self.input_len, 300)
        df = get_klines(symbol, "1m", limit=need+5)
        if len(df) < self.input_len:
            return None, 0.0, {"reason":"short_klines"}
        feats = _build_features(df)
        X = feats[FEATURE_COLS].to_numpy(dtype=np.float32)[-self.input_len:]
        mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-8
        Xn = (X - mu)/sd
        x = torch.tensor(Xn, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_pred, c_logit = self.model(x)
        q_pred = q_pred.view(1, len(self.horizons), len(self.quantiles)).cpu().numpy()
        h = min(self.use_h_idx, q_pred.shape[1]-1)
        q10, q50, q90 = float(q_pred[0,h,0]), float(q_pred[0,h,1]), float(q_pred[0,h,2])
        p_tp = torch.sigmoid(c_logit.squeeze(-1)).item()
        tp_bps, sl_bps = self.tp_bps, self.sl_bps
        EV = p_tp*tp_bps - (1.0-p_tp)*sl_bps - self.cost_bps
        q50_bps = 1e4*q50; band_bps = 1e4*(q90-q10)
        enter = (EV >= self.theta_bps) and (q50_bps >= self.cost_bps+5) and (band_bps <= self.band_max) and (q10 > -sl_bps/2/1e4)
        side = "Buy" if q50 >= 0 else "Sell"
        conf = 0.55 + max(0.0, EV)/max(tp_bps+sl_bps,1e-9)*0.4
        conf = float(min(0.99, conf))
        info = {"p_tp":p_tp, "q10":q10, "q50":q50, "q90":q90, "EV":EV, "q50_bps":q50_bps, "band_bps":band_bps, "enter":bool(enter), "horizon":self.horizons[h], "head":self.model._head_type}
        if not enter:
            return None, 0.0, info
        return side, conf, info

# =========================
# Tick bus
# =========================
class TickBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._evt  = threading.Event()
        self.snapshot: Dict[str, Tuple[float,float,float,float]] = {}
        self.tick_id = 0
        self.ts = 0.0
        self._stop = False
        self._last_hb = 0.0
    def start(self):
        t = threading.Thread(target=self._run, daemon=True); t.start()
        try:
            _ = get_all_book_tickers()
        except Exception:
            pass
    def stop(self): self._stop = True
    def wait_tick(self, last_seen: int, timeout: float=5.0) -> Optional[int]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.tick_id > last_seen: return self.tick_id
            self._evt.wait(0.05)
        return None
    def _run(self):
        while not self._stop:
            try:
                snap = get_all_book_tickers()
                if snap:
                    with self._lock:
                        self.snapshot = snap
                        self.ts = time.time()
                        self.tick_id += 1
                    self._evt.set(); self._evt.clear()
                    if VERBOSE and (self.ts - self._last_hb) >= HEARTBEAT_SEC:
                        self._last_hb = self.ts
                        print("\t")
            except Exception:
                pass
            time.sleep(TICK_INTERVAL_SEC)
    def get(self) -> Tuple[int, float, Dict[str, Tuple[float,float,float,float]]]:
        with self._lock:
            return self.tick_id, self.ts, dict(self.snapshot)

# =========================
# 백업 시그널
# =========================
def _ma_side(symbol: str) -> Tuple[Optional[str], float]:
    try:
        df = get_klines(symbol, "1m", 120)
        if len(df) < 60: return None, 0.0
        closes = df["close"].values
        ma20 = float(np.mean(closes[-20:])); ma60 = float(np.mean(closes[-60:]))
        if ma20>ma60: return "Buy", 0.60
        if ma20<ma60: return "Sell", 0.60
        return None, 0.0
    except Exception:
        return None, 0.0

def _opp(side: str) -> str: return "Sell" if side=="Buy" else "Buy"

# =========================
# Strategy glue
# =========================
_PATCH_RUNNER: Optional[PatchTSTRunner] = None
def _get_runner() -> Optional[PatchTSTRunner]:
    global _PATCH_RUNNER
    if _PATCH_RUNNER is None:
        try:
            _PATCH_RUNNER = PatchTSTRunner()
        except Exception as e:
            if VERBOSE:
                print(f"[PatchTST][INIT][ERR] {e}")
            _PATCH_RUNNER = None
    return _PATCH_RUNNER

def three_model_consensus(symbol: str):
    runner = _get_runner()
    if runner:
        side, conf, info = runner.predict_symbol(symbol)
        if side:
            return side, [conf], info, None
    side2, conf2 = _ma_side(symbol)
    return side2, [conf2], {"fallback":"ma"}, None

def choose_entry(symbol: str) -> Tuple[Optional[str], float]:
    try:
        side_cons, confs, _, _ = three_model_consensus(symbol)
        if side_cons in ("Buy","Sell"):
            conf = float(max([c for c in (confs or []) if isinstance(c,(int,float))] or [0.6]))
            return side_cons, conf
    except Exception:
        pass
    return _ma_side(symbol)

def apply_mode(side: Optional[str]) -> Optional[str]:
    if side not in ("Buy","Sell"): return None
    m = (ENTRY_MODE or "model").lower()
    if m=="model": return side
    if m=="inverse": return _opp(side)
    if m=="random": return "Buy" if (time.time()*1000)%2<1 else "Sell"
    return side

# =========================
# Broker (paper)
# =========================
class PaperBroker:
    def __init__(self, starting_cash: float, leverage: int, tag: str):
        self.cash = float(starting_cash); self.lev  = int(leverage); self.tag  = tag
        self.pos: Dict[str, dict] = {}; self.cool: Dict[str, float] = defaultdict(float)
        self.last_eq_ts = 0.0
        self.trades_csv = os.path.join(OUT_DIR, f"trades_{tag}_bybit.csv")
        self.equity_csv = os.path.join(OUT_DIR, f"equity_{tag}_bybit.csv")
        self._init_csv()
    def _init_csv(self):
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","pnl","roi","cash","eq","hold"])
        if not os.path.exists(self.equity_csv):
            with open(self.equity_csv,"w",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(["ts","equity","cash","upnl"])
    def _wtrade(self, row: dict):
        with open(self.trades_csv,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k,"") for k in ["ts","event","symbol","side","qty","entry","exit","tp1","tp2","tp3","sl","pnl","roi","cash","eq","hold"]])
    def _weq(self, ts: float, eq: float, upnl: float):
        if time.time() - self.last_eq_ts < 3.0: return
        with open(self.equity_csv,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([int(ts), f"{eq:.6f}", f"{self.cash:.6f}", f"{upnl:+.6f}"])
        self.last_eq_ts = time.time()
    def can_open(self) -> bool: return len(self.pos) < MAX_OPEN
    def equity(self, mids: Dict[str,float]) -> Tuple[float, float]:
        up = 0.0
        for sym, p in self.pos.items():
            mid = mids.get(sym)
            if mid is None: continue
            up += p["qty"]*((mid - p["entry"]) if p["side"]=="Buy" else (p["entry"] - mid))
        return self.cash + up, up
    def try_open(self, sym: str, side: str, mid: float, ts: float):
        if sym in self.pos or not self.can_open(): return
        eq,_ = self.equity({})
        free_cash = self.cash
        use_cap   = eq * (RISK_PCT_OF_EQUITY/1.0)
        use_cash  = min(free_cash*ENTRY_PORTION, use_cap)
        if use_cash <= 0 or mid <= 0: return
        notional_raw = use_cash * self.lev
        notional_cap = min(MAX_NOTIONAL_ABS, eq * (MAX_NOTIONAL_PCT_OF_EQUITY/100.0))
        notional     = max(MIN_NOTIONAL_USDT, min(notional_raw, notional_cap))
        rule = rule_of(sym)
        slip = SLIP_BPS_IN/10000.0
        entry = mid*(1.0 + slip if side=="Buy" else 1.0 - slip)
        qty = notional / max(entry,1e-12)
        qty = math.floor(qty/rule["step"])*rule["step"]
        if qty < rule["min"]: qty = rule["min"]
        tp1 = entry*(1.0 + TP1_BPS/10000.0) if side=="Buy" else entry*(1.0 - TP1_BPS/10000.0)
        tp2 = entry*(1.0 + TP2_BPS/10000.0) if side=="Buy" else entry*(1.0 - TP2_BPS/10000.0)
        tp3 = entry*(1.0 + TP3_BPS/10000.0) if side=="Buy" else entry*(1.0 - TP3_BPS/10000.0)
        sl  = entry*(1.0 - SL_ROI_PCT) if side=="Buy" else entry*(1.0 + SL_ROI_PCT)
        self.cash -= notional*TAKER_FEE
        self.pos[sym] = {"side":side,"qty":qty,"entry":entry,"tp":[tp1,tp2,tp3],
                         "tp_done":[False,False,False],"tp_ratio":[TP1_RATIO,TP2_RATIO,1.0],
                         "sl":sl,"be":False,"entry_ts":ts}
        eq_now,_ = self.equity({sym:mid})
        self._wtrade({"ts":int(ts),"event":"ENTRY","symbol":sym,"side":side,"qty":f"{qty:.8f}","entry":f"{entry:.6f}",
                      "exit":"","tp1":f"{tp1:.6f}","tp2":f"{tp2:.6f}","tp3":f"{tp3:.6f}","sl":f"{sl:.6f}",
                      "pnl":"","roi":"","cash":f"{self.cash:.6f}","eq":f"{eq_now:.6f}","hold":""})
    def _partial_exit(self, sym: str, px: float, ratio: float, reason: str, ts: float):
        p = self.pos[sym]; side, entry, qty0 = p["side"], p["entry"], p["qty"]
        close_qty = max(0.0, min(qty0, qty0*ratio))
        if close_qty <= 0: return
        slip = SLIP_BPS_OUT/10000.0
        exit_px = px*(1.0 - slip) if side=="Buy" else px*(1.0 + slip)
        pnl = close_qty*((exit_px-entry) if side=="Buy" else (entry-exit_px))
        fee = close_qty*exit_px*TAKER_FEE
        pnl_after = pnl - fee
        self.cash += pnl_after
        p["qty"] = float(qty0 - close_qty)
        hold = max(0.0, ts - p["entry_ts"])
        eq_now,_ = self.equity({sym:px})
        self._wtrade({"ts":int(ts),"event":"EXIT_PARTIAL","symbol":sym,"side":side,"qty":f"{close_qty:.8f}",
                      "entry":f"{entry:.6f}","exit":f"{exit_px:.6f}","tp1":"","tp2":"","tp3":"",
                      "sl":f"{p.get('sl',0):.6f}","pnl":f"{pnl_after:.6f}","roi":f"{_roi_pct(side,entry,exit_px):.6f}",
                      "cash":f"{self.cash:.6f}","eq":f"{eq_now:.6f}","hold":f"{hold:.2f}"})
        if p["qty"] <= 0: del self.pos[sym]
    def close_all(self, sym: str, px: float, reason: str, ts: float):
        if sym not in self.pos: return
        self._partial_exit(sym, px, 1.0, reason, ts)
    def on_tick(self, tick_ts: float, mids: Dict[str,float], bids: Dict[str,float], asks: Dict[str,float]):
        eq, up = self.equity(mids)
        self._weq(tick_ts, eq, up)
        to_close = []
        for sym, p in list(self.pos.items()):
            if sym not in mids: continue
            mid = mids[sym]; bid=bids[sym]; ask=asks[sym]; side = p["side"]; qty=p["qty"]; entry=p["entry"]
            for i, tp_px in enumerate(p["tp"]):
                if p["tp_done"][i] or qty<=0: continue
                hit = (mid>=tp_px) if side=="Buy" else (mid<=tp_px)
                if hit:
                    px_exec = bid if side=="Buy" else ask
                    self._partial_exit(sym, px_exec, p["tp_ratio"][i], f"TP{i+1}", tick_ts)
                    p = self.pos.get(sym);
                    if not p: break
                    p["tp_done"][i]=True; qty = p["qty"]
                    if (i+1)>=BE_AFTER_TIER and not p["be"]:
                        be_eps = BE_EPS_BPS/10000.0
                        p["sl"] = entry*(1.0 + be_eps) if side=="Buy" else entry*(1.0 - be_eps)
                        p["be"]=True
            if sym not in self.pos: continue
            tiers_done = sum(1 for x in p["tp_done"] if x)
            if tiers_done >= TRAIL_AFTER_TIER:
                if side=="Buy":
                    trail = mid*(1.0 - TRAIL_BPS/10000.0)
                    if p["be"]: trail = max(trail, entry*(1.0 + BE_EPS_BPS/10000.0))
                    if trail > p["sl"]: p["sl"]=trail
                else:
                    trail = mid*(1.0 + TRAIL_BPS/10000.0)
                    if p["be"]: trail = min(trail, entry*(1.0 - BE_EPS_BPS/10000.0))
                    if p["sl"]==0 or trail < p["sl"]: p["sl"]=trail
            sl_now = p.get("sl", 0.0) or 0.0
            if sl_now>0:
                if (side=="Buy" and bid <= sl_now) or (side=="Sell" and ask >= sl_now):
                    px_exec = bid if side=="Buy" else ask
                    to_close.append((sym, px_exec, "SL"))
            if MAX_HOLD_SEC and (tick_ts - p["entry_ts"]) >= MAX_HOLD_SEC:
                to_close.append((sym, mid, "TIMEOUT"))
        for sym, px, reason in to_close:
            self.close_all(sym, px, reason, tick_ts)
            self.cool[sym] = time.time() + COOLDOWN_SEC

# =========================
# Worker
# =========================
class ScenarioWorker(threading.Thread):
    def __init__(self, bus: "TickBus", leverage: int, start_cash: float):
        tag = f"lev{leverage}_cash{int(start_cash)}"
        super().__init__(daemon=True, name=f"Worker-{tag}")
        self.bus = bus
        self.tag = tag
        self.broker = PaperBroker(start_cash, leverage, tag)
        self._stop = False
        self._last_tick = 0
    def stop(self): self._stop = True
    def run(self):
        printed_zero = False
        while not self._stop:
            tid = self.bus.wait_tick(self._last_tick, timeout=5.0)
            if tid is None:
                if VERBOSE and not printed_zero: printed_zero = True
                continue
            printed_zero = False
            self._last_tick = tid
            tid2, ts, snap = self.bus.get()
            if tid2 != tid: continue
            bids={}; asks={}; mids={}; marks={}
            for s,(b,a,m,mark) in snap.items():
                bids[s]=b; asks[s]=a; mids[s]=m; marks[s]=mark
            self.broker.on_tick(ts, mids, bids, asks)
            opened = 0
            for sym in SYMBOLS:
                if sym not in mids: continue
                if self.broker.cool.get(sym,0.0) > time.time(): continue
                if sym in self.broker.pos: continue
                if not self.broker.can_open(): break
                side, conf = choose_entry(sym)
                side = apply_mode(side)
                if side not in ("Buy","Sell") or conf < 0.55: continue
                if USE_MARK_PRICE_FOR_RISK:
                    try:
                        mark = marks.get(sym) or get_mark_price(sym)
                        if mark>0 and abs(mark - mids[sym])/max(mids[sym],1e-9) > MAX_MARK_FAIR_DIFF:
                            continue
                    except Exception:
                        pass
                self.broker.try_open(sym, side, mids[sym], ts)
                opened += 1
            if VERBOSE and (tid % WORKER_STATUS_EVERY == 0):
                pos_cnt = len(self.broker.pos)
                eq,_ = self.broker.equity(mids)
                print(f"[{self.tag}] tick={tid} pos={pos_cnt} opened={opened} eq={eq:.2f}")

# =========================
# Main
# =========================
def main():
    print(f"[START] {len(LEVERAGE_LIST)*len(START_CASH_LIST)} scenarios | symbols={len(SYMBOLS)} | interval={TICK_INTERVAL_SEC}s")
    print(f"ENTRY_MODE={ENTRY_MODE}  TP={TP1_BPS}/{TP2_BPS}/{TP3_BPS}bps  SL={SL_ROI_PCT*100:.2f}%")
    bus = TickBus(); bus.start()
    workers: List[ScenarioWorker] = []
    for lev in LEVERAGE_LIST:
        for cash in START_CASH_LIST:
            workers.append(ScenarioWorker(bus, lev, cash))
    for w in workers: w.start()
    stop_flag = threading.Event()
    def _sigint(_sig,_frm): stop_flag.set()
    signal.signal(signal.SIGINT, _sigint)
    try:
        while not stop_flag.is_set():
            time.sleep(0.5)
    finally:
        for w in workers: w.stop()
        bus.stop()
        print("[STOP] done")

if __name__ == "__main__":
    main()
