# train_patchtst_bybit.py
# -*- coding: utf-8 -*-
"""
Bybit 1m OHLCV → PatchTST 멀티태스크 학습 스크립트
- 입력: bybit_futures_1m.parquet (또는 CSV: bybit_futures_1m.csv)
- 목표:
  (1) 멀티-호라이즌 분포예측(Quantile: 0.1/0.5/0.9) — 수익률 회귀
  (2) TP-first 확률 분류(주어진 tp_bps, sl_bps에서 TP가 SL보다 먼저 도달하는지)
- 의존: pip install torch pandas numpy
- 실행:
    python train_patchtst_bybit.py --data bybit_futures_1m.parquet --epochs 10 --device cuda
"""
import os, math, argparse, random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    input_len: int = 240
    horizons: Tuple[int, ...] = (1, 5, 15, 60)
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    tp_bps: int = 120
    sl_bps: int = 90
    patch_len: int = 32
    patch_stride: int = 16
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 2048
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    num_workers: int = 8
    seed: int = 42
    alpha_q: float = 1.0
    alpha_cls_start: float = 1.0
    alpha_cls_end: float = 4.0
    use_symbol_norm: bool = True
    val_ratio: float = 0.1
    # ↓ 추가
    use_amp: bool = True
    use_compile: bool = True


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_ohlcv(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("input must have 'timestamp' column")

    # 어떤 형식이든 UTC aware로 통일
    ts = df["timestamp"]
    import pandas.api.types as pdt

    if pdt.is_datetime64_any_dtype(ts) or pdt.is_datetime64tz_dtype(ts):
        # 이미 datetime인 경우
        if getattr(ts.dt, "tz", None) is None:
            df["timestamp"] = ts.dt.tz_localize("UTC")
        else:
            df["timestamp"] = ts.dt.tz_convert("UTC")
    else:
        # 숫자/문자 → datetime(UTC) 변환
        # 밀리초/초 자동 판별
        s = pd.to_numeric(ts, errors="coerce")
        if s.notna().any():
            # 값 범위로 ms/초 추정
            is_ms = (s.dropna().median() > 1e11)
            unit = "ms" if is_ms else "s"
            df["timestamp"] = pd.to_datetime(s, unit=unit, utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # 심볼별 파생특징: log-return, 누적수익, RV, 가격/거래량 지표
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        p = g["close"].astype(float).replace(0, np.nan).ffill()
        logp = np.log(p)
        r1 = logp.diff()                        # 1분 log-return
        r5 = logp.diff(5)
        r15 = logp.diff(15)
        rv5 = (r1**2).rolling(5).sum()
        rv15 = (r1**2).rolling(15).sum()
        vol = g["volume"].astype(float).fillna(0.0)
        v5 = vol.rolling(5).mean()
        v15 = vol.rolling(15).mean()
        spr = (g["high"] - g["low"]).astype(float) / g["close"].replace(0,np.nan)
        spr = spr.replace([np.inf,-np.inf], np.nan).fillna(0.0)
        g["f_ret1"] = r1.fillna(0.0)
        g["f_ret5"] = r5.fillna(0.0)
        g["f_ret15"] = r15.fillna(0.0)
        g["f_rv5"] = rv5.fillna(0.0)
        g["f_rv15"] = rv15.fillna(0.0)
        g["f_v5"] = v5.fillna(0.0)
        g["f_v15"] = v15.fillna(0.0)
        g["f_spr"] = spr
        g["price"] = p.values
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res

def make_targets(df: pd.DataFrame, horizons: Tuple[int,...]) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        p = g["price"].astype(float).values
        for h in horizons:
            # h분 수익률: (P_{t+h} - P_t)/P_t
            g[f"y_ret_{h}"] = (pd.Series(p).shift(-h) - p) / p
        out.append(g)
    return pd.concat(out, ignore_index=True)

def label_tp_first_segment(price: np.ndarray, tp_bps=80, sl_bps=80, horizon=60) -> np.ndarray:
    """
    TP가 SL보다 먼저 맞는지(1/0). 아무 것도 맞지 않으면 np.nan
    """
    n = len(price)
    y = np.full(n, np.nan, dtype=float)
    up = price * (1.0 + tp_bps/1e4)
    dn = price * (1.0 - sl_bps/1e4)
    for i in range(n-horizon):
        seg = price[i+1:i+1+horizon]
        hit_tp = np.where(seg >= up[i])[0]
        hit_sl = np.where(seg <= dn[i])[0]
        t_tp = hit_tp[0] if hit_tp.size else np.inf
        t_sl = hit_sl[0] if hit_sl.size else np.inf
        if t_tp==np.inf and t_sl==np.inf:
            y[i] = np.nan
        else:
            y[i] = 1.0 if t_tp < t_sl else 0.0
    return y

def add_tp_first(df: pd.DataFrame, tp_bps=80, sl_bps=80, horizon=60) -> pd.DataFrame:
    out=[]
    for sym, g in df.groupby("symbol", sort=False):
        g=g.copy()
        g["y_tpfirst"] = label_tp_first_segment(g["price"].values, tp_bps, sl_bps, horizon)
        out.append(g)
    return pd.concat(out, ignore_index=True)

def train_val_split_indices(df: pd.DataFrame, val_ratio: float) -> Tuple[np.ndarray,np.ndarray]:
    # 시계열 분할: 각 심볼별 마지막 val_ratio 비율을 검증으로
    train_idx = []
    val_idx = []
    for sym, g in df.groupby("symbol", sort=False):
        n = len(g)
        split = int(n * (1.0 - val_ratio))
        gi = g.index.values
        train_idx.append(gi[:split])
        val_idx.append(gi[split:])
    return np.concatenate(train_idx), np.concatenate(val_idx)

# -------------------------
# Dataset
# -------------------------
FEATURE_COLS = ["f_ret1","f_ret5","f_ret15","f_rv5","f_rv15","f_v5","f_v15","f_spr"]

class PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices: np.ndarray, cfg: CFG):
        self.df = df
        self.idx = indices
        self.cfg = cfg
        self.syms = df["symbol"].values
        # 정규화 파라미터(심볼별)
        if cfg.use_symbol_norm:
            self.sym_stats = self._compute_symbol_stats()
        else:
            self.sym_stats = None

    def _compute_symbol_stats(self):
        d = {}
        for sym, g in self.df.groupby("symbol", sort=False):
            X = g[FEATURE_COLS].values.astype(np.float32)
            mu = X.mean(axis=0)
            sd = X.std(axis=0) + 1e-8
            d[sym] = (mu, sd)
        return d

    def __len__(self):
        # 윈도 끝에 라벨 필요: max(horizons)
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i]
        L = self.cfg.input_len
        Hmax = max(self.cfg.horizons)
        start = idx - L + 1
        end = idx + 1
        # 경계 보호
        if start < 0:
            start = 0
            end = L
        # 심볼 경계 넘어가지 않도록 보호
        sym = self.df.iloc[idx]["symbol"]
        g = self.df[self.df["symbol"]==sym]
        gi = g.index.values
        # 현재 idx가 포함된 위치 찾기
        if idx not in gi:
            raise IndexError("symbol boundary issue")
        pos = np.where(gi == idx)[0][0]
        if pos < L-1 or pos+Hmax >= len(gi):
            # 샘플 불가 → 다음 항목으로 대체(러프 처리)
            return self.__getitem__((i+1) % len(self.idx))
        window_idx = gi[pos-L+1:pos+1]
        # 입력 X
        X = self.df.loc[window_idx, FEATURE_COLS].values.astype(np.float32)
        # 정규화
        if self.sym_stats is not None:
            mu, sd = self.sym_stats[sym]
            X = (X - mu.astype(np.float32)) / sd.astype(np.float32)
        # 라벨 Y (연속 수익률, shape = n_h)
        y_list = []
        for h in self.cfg.horizons:
            y = self.df.loc[gi[pos], f"y_ret_{h}"]
            y_list.append(np.float32(y))
        y = np.array(y_list, dtype=np.float32)  # [n_h]
        # TP-first 라벨 및 마스크
        y_tp = self.df.loc[gi[pos], "y_tpfirst"]
        mask_tp = 0.0 if np.isnan(y_tp) else 1.0
        y_tp = 0.0 if np.isnan(y_tp) else float(y_tp)
        return torch.from_numpy(X), torch.from_numpy(y), torch.tensor(y_tp, dtype=torch.float32), torch.tensor(mask_tp, dtype=torch.float32)

# -------------------------
# Model: PatchTST (간소)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d]

    def forward(self, x):
        # x: [B, L, d]
        L = x.size(1)
        return x + self.pe[:, :L, :]

class PatchTST(nn.Module):
    def __init__(self, n_features:int, cfg:CFG):
        super().__init__()
        self.cfg = cfg
        P, S = cfg.patch_len, cfg.patch_stride
        # 패치 개수
        self.n_patches = 1 + (cfg.input_len - P) // S
        self.patch_dim = P * n_features
        self.embed = nn.Linear(self.patch_dim, cfg.d_model)
        self.posenc = PositionalEncoding(cfg.d_model, max_len=self.n_patches)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model*4, dropout=cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        # 토큰 풀링
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 헤드A: Quantile (n_h * n_q 출력)
        self.head_q = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, len(cfg.horizons)*len(cfg.quantiles))
        )
        # 헤드B: TP-first 분류(시그모이드)
        self.head_cls = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model//2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model//2, 1)
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F] → patches: [B, T, (P*F)]
        """
        B,L,F = x.shape
        P,S = self.cfg.patch_len, self.cfg.patch_stride
        T = 1 + (L - P)//S
        xs = []
        for t in range(T):
            s = t*S
            xs.append(x[:, s:s+P, :].reshape(B, -1))
        return torch.stack(xs, dim=1)

    def forward(self, x: torch.Tensor):
        # x: [B, L, F]
        patches = self._patchify(x)                # [B, T, P*F]
        z = self.embed(patches)                    # [B, T, d]
        z = self.posenc(z)                         # [B, T, d]
        z = self.encoder(z)                        # [B, T, d]
        # pool over tokens
        z_pooled = self.pool(z.transpose(1,2)).squeeze(-1)  # [B, d]
        # heads
        q = self.head_q(z_pooled)                  # [B, n_h*n_q]
        c = self.head_cls(z_pooled)                # [B, 1]
        return q, c

# -------------------------
# Losses & Metrics
# -------------------------
def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    pred: [B, n_h*n_q], target: [B, n_h]
    """
    B = target.size(0)
    n_h = target.size(1)
    n_q = len(quantiles)
    pred = pred.view(B, n_h, n_q)
    t = target.unsqueeze(-1).expand_as(pred)
    loss = 0.0
    for qi, q in enumerate(quantiles):
        e = t[:,:,qi] - pred[:,:,qi]
        loss_q = torch.maximum(q*e, (q-1)*e).mean()
        loss = loss + loss_q
    return loss / n_q

def bce_masked(pred_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred_logits: [B,1], target: [B], mask: [B] (0=ignore)
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_logits.device)
    loss = nn.functional.binary_cross_entropy_with_logits(pred_logits.squeeze(-1)[mask>0.5], target[mask>0.5])
    return loss

# -------------------------
# Train
# -------------------------
def make_loaders(df: pd.DataFrame, cfg: CFG):
    # 유효 라벨 필터: 모든 y_ret_h가 존재해야 함
    for h in cfg.horizons:
        df = df[~df[f"y_ret_{h}"].isna()]
    train_idx, val_idx = train_val_split_indices(df, cfg.val_ratio)

    # 학습 가능한 인덱스는 input_len 이상 경과한 지점만
    train_idx = train_idx[train_idx >= df.index.min()+cfg.input_len+max(cfg.horizons)]
    val_idx = val_idx[val_idx >= df.index.min()+cfg.input_len+max(cfg.horizons)]

    tr_ds = PatchDataset(df, train_idx, cfg)
    va_ds = PatchDataset(df, val_idx, cfg)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    return tr_dl, va_dl

# ---- Loss: Focal BCE with logits (마스크 지원) ----
# ---- Focal BCE with logits (AMP 안전) ----
def focal_bce_with_logits(logits, targets, mask, alpha=0.5, gamma=2.0):
    m = mask > 0.5
    if m.sum() == 0:
        return logits.new_tensor(0.0)
    z = logits.squeeze(-1)[m]                  # logits
    y = targets[m].to(z.dtype)                 # 0/1, float
    # BCE with logits → AMP 안전
    ce = nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
    p  = torch.sigmoid(z)                      # prob for focal modulator
    pt = torch.where(y == 1, p, 1 - p)
    return (alpha * (1 - pt)**gamma * ce).mean()

#from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, dl, device, cfg):
    model.eval()
    tot_q=tot_c=n_q=n_c=0
    ys=[]; ps=[]
    with torch.no_grad():
        for X,y,ytp,m in dl:
            X=X.to(device, non_blocking=True); y=y.to(device)
            ytp=ytp.to(device); m=m.to(device)
            q,c = model(X)
            lq = pinball_loss(q,y,list(cfg.quantiles))
            lc = focal_bce_with_logits(c, ytp, m, alpha=0.5, gamma=2.0)
            bs = X.size(0)
            tot_q += lq.item()*bs; n_q += bs
            valid = (m>0.5)
            v = valid.sum().item()
            if v>0:
                tot_c += lc.item()*v; n_c += v
                p = torch.sigmoid(c.squeeze(-1)[valid]).cpu().numpy()
                t = ytp[valid].cpu().numpy()
                ys.append(t); ps.append(p)
    vp = tot_q/max(n_q,1)
    vc = tot_c/max(n_c,1) if n_c>0 else 0.0
    auc = ap = float("nan")
    if ys:
        y = np.concatenate(ys); p = np.concatenate(ps)
        if y.min()!=y.max(): auc = roc_auc_score(y,p)
        if y.sum()>0:        ap  = average_precision_score(y,p)
        pos_rate = float(y.mean())
    else:
        pos_rate = float("nan")
    return vp, vc, auc, ap, pos_rate

def train_loop(model, tr_dl, va_dl, device, cfg: CFG, outdir="ckpt"):
    os.makedirs(outdir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
    best = 1e9; best_path=None

    for ep in range(1, cfg.epochs+1):
        model.train()
        # 램프 완화: 1 → 2 (5에폭 내)
        ramp = min(1.0, (ep-1)/4.0)
        alpha_cls = 1.0 + (2.0 - 1.0)*ramp

        tot_q = 0.0; n_q = 0
        tot_c = 0.0; n_c = 0

        for X, y, ytp, m in tr_dl:
            X = X.to(device, non_blocking=True)
            y, ytp, m = y.to(device), ytp.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                q, c = model(X)
                lq = pinball_loss(q, y, list(cfg.quantiles))
                lc = focal_bce_with_logits(c, ytp, m, alpha=0.5, gamma=2.0)
                # 손실 결합
                loss = cfg.alpha_q*lq + alpha_cls*lc
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            bs = X.size(0)
            tot_q += lq.item()*bs; n_q += bs
            valid = (m > 0.5).sum().item()
            if valid>0:
                tot_c += lc.item()*valid; n_c += valid

        tr_q = tot_q/max(n_q,1)
        tr_c = tot_c/max(n_c,1) if n_c>0 else 0.0

        va_q, va_c, auc, ap, pos_rate = evaluate(model, va_dl, device, cfg)
        va_total = cfg.alpha_q*va_q + alpha_cls*va_c
        sched.step(va_total)
        print(f"[E{ep:02d}] train pinball={tr_q:.6f} cls={tr_c:.6f} | "
              f"val pinball={va_q:.6f} cls={va_c:.6f} total={va_total:.6f} | "
              f"AUC={auc:.3f} AP={ap:.3f} pos={pos_rate:.2f}")

        if va_total < best:
            best = va_total
            best_path = os.path.join(outdir, f"patchtst_best_ep{ep}.pt")
            torch.save({"model":model.state_dict(),"cfg":cfg.__dict__}, best_path)

    print(f"[BEST] {best_path} total_loss={best:.6f}")
    return best_path

# -------------------------
# Inference helper
# -------------------------
def load_best(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

def decide_from_batch(model, feats_np: np.ndarray, cfg: CFG, cost_bps: float=8.0, thresh_bps: float=10.0):
    """
    feats_np: [L, F] 최근 L=input_len 분 윈도우(정규화 전)
    간단화를 위해 정규화는 호출 전 동일 방식으로 해야 정확. 여기서는 무정규화 입력 가정시 참고용.
    """
    x = torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0)  # [1,L,F]
    with torch.no_grad():
        q_pred, c_logit = model(x)
        q_pred = q_pred.view(1, len(cfg.horizons), len(cfg.quantiles))
        q50 = q_pred[0,:,1].cpu().numpy()  # Q0.5
        p_tp = torch.sigmoid(c_logit.squeeze(-1)).item()
    tp_bps = cfg.tp_bps; sl_bps = cfg.sl_bps
    ev_bps = p_tp*tp_bps - (1.0-p_tp)*sl_bps - cost_bps
    side = "Buy" if q50[0] >= 0 else "Sell"
    enter = (ev_bps >= thresh_bps) and (q50[0]*1e4 >= cost_bps+5)
    return dict(enter=bool(enter), side=side, p_tp=float(p_tp), q50=list(q50), ev_bps=float(ev_bps))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="bybit_futures_1m.parquet")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=CFG.batch_size)
    args = ap.parse_args()

    cfg = CFG(epochs=args.epochs, batch_size=args.batch)
    set_seed(cfg.seed)

    print("[LOAD]", args.data)
    df = load_ohlcv(args.data)
    # 필수 컬럼 검사
    for col in ["symbol","close","high","low","volume"]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    print("[FEATS] build features/targets")
    df = add_features(df)
    df = make_targets(df, cfg.horizons)
    df = add_tp_first(df, cfg.tp_bps, cfg.sl_bps, max(cfg.horizons))

    # 학습/검증 로더
    tr_dl, va_dl = make_loaders(df, cfg)

    device = torch.device(args.device)
    n_features = len(FEATURE_COLS)
    model = PatchTST(n_features, cfg).to(device)
    print("CUDA avail:", torch.cuda.is_available())
    print("GPU count :", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name  :", torch.cuda.get_device_name(0))
        print("CUDA ver  :", torch.version.cuda)
        print("Torch CUDA:", torch.backends.cudnn.version())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTST(n_features, cfg).to(device)
    print("Model device:", next(model.parameters()).device)
    print("[TRAIN] start")
    best_path = train_loop(model, tr_dl, va_dl, device, cfg, outdir="ckpt")

    print("[DONE]", best_path)

if __name__ == "__main__":
    main()
