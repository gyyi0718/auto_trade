# tcn_train_crypto_dir.py
# -*- coding: utf-8 -*-
"""
TCN for multi-symbol 1m futures
- 입력: bybit_futures_1m.csv/parquet (timestamp,symbol,open,high,low,close,volume,turnover)
- 타깃: H분 후 로그수익률 y = log(close[t+H]/close[t])
- 손실:
    LOSS_MODE=cls   -> BCE(방향 전용)
    LOSS_MODE=joint -> Huber(회귀) + λ*BCE(부호)
- 노이즈 데드존: EPS_BPS로 |y|가 작은 샘플의 BCE 가중치 축소/마스킹
Env:
  DATA_PATH=bybit_futures_1m.parquet|csv
  H=3
  SEQ_LEN=60
  BATCH=512
  EPOCHS=30
  LR=1e-3
  TRAIN_DAYS=30
  VAL_DAYS=30
  SYM_KEEP=ETHUSDT,BTCUSDT,...
  LOSS_MODE=cls|joint
  BCE_LAMBDA=1.0        # joint일 때 가중
  EPS_BPS=3.0           # |y|<EPS_BPS bps는 BCE 가중치↓
  BCE_MASK_SMALL=0|1    # 1이면 작은 |y| 샘플 BCE 완전 제외
"""

import os, math, numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import weight_norm

# --------------------
# Config
# --------------------
DATA_PATH = os.getenv("DATA_PATH", "bybit_futures_1m.parquet")
H         = int(os.getenv("H", "3"))
SEQ_LEN   = int(os.getenv("SEQ_LEN", "240"))
BATCH     = int(os.getenv("BATCH", "4096"))
EPOCHS    = int(os.getenv("EPOCHS", "30"))
LR        = float(os.getenv("LR", "1e-4"))
TRAIN_DAYS= int(os.getenv("TRAIN_DAYS", "700"))
VAL_DAYS  = int(os.getenv("VAL_DAYS", "20"))
SYM_KEEP  = [s for s in os.getenv("SYM_KEEP","").split(",") if s.strip()]

LOSS_MODE = os.getenv("LOSS_MODE","cls").lower()  # cls | joint
BCE_LAMBDA = float(os.getenv("BCE_LAMBDA","1.0"))
EPS_BPS = float(os.getenv("EPS_BPS", "30.0"))  # |tgt| < EPS_BPS(bps) → 학습 제외
BCE_MASK_SMALL = int(os.getenv("BCE_MASK_SMALL","1"))  # 1=|y|<eps 샘플 BCE 제외

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# I/O
# --------------------
def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    need = {"timestamp","symbol","close","open","high","low","volume","turnover"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    if SYM_KEEP:
        df = df[df["symbol"].isin(SYM_KEEP)].reset_index(drop=True)
    return df

# --------------------
# Feature engineering
# --------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        c = g["close"].astype(float).values
        v = g["volume"].astype(float).values

        ret = np.zeros_like(c, dtype=np.float64)
        ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))

        win = 60
        rv = pd.Series(ret).rolling(win, min_periods=win//3).std().bfill().values

        ema_win = 60
        alpha = 2/(ema_win+1)
        ema = np.zeros_like(c, dtype=np.float64); ema[0]=c[0]
        for i in range(1,len(c)): ema[i] = alpha*c[i] + (1-alpha)*ema[i-1]
        mom = (c/np.maximum(ema,1e-12) - 1.0)

        v_ma = pd.Series(v).rolling(win, min_periods=win//3).mean().bfill().values
        v_sd = pd.Series(v).rolling(win, min_periods=win//3).std().replace(0, np.nan).bfill().values
        vz  = (v - v_ma) / np.where(v_sd>0, v_sd, 1.0)

        g["ret"] = ret; g["rv"]=rv; g["mom"]=mom; g["vz"]=vz
        out.append(g)
    df2 = pd.concat(out, ignore_index=True)
    df2["tgt"] = df2.groupby("symbol")["close"].transform(lambda s: np.log(s.shift(-H) / s))
    return df2

# --------------------
# Scaling
# --------------------
FEATS = ["ret","rv","mom","vz"]

def fit_scaler(df: pd.DataFrame, cols):
    mu = df[cols].mean().values
    sd = df[cols].std(ddof=0).replace(0, np.nan).fillna(1.0).values if isinstance(df[cols].std(), pd.Series) else df[cols].std(ddof=0).values
    sd = np.where(sd>0, sd, 1.0)
    return mu, sd

def apply_scaler(df: pd.DataFrame, cols, mu, sd):
    df = df.copy()
    df[cols] = (df[cols].values - mu) / sd
    return df

# --------------------
# Split
# --------------------
def split_time(df: pd.DataFrame):
    tmax = df["timestamp"].max()
    val_start = tmax - pd.Timedelta(days=VAL_DAYS)
    train_start = val_start - pd.Timedelta(days=TRAIN_DAYS)
    train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < val_start)].copy()
    val   = df[(df["timestamp"] >= val_start)].copy()
    return train, val

class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=2.0, reduction='none'):
        super().__init__(); self.gamma=gamma; self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        return ( (1-pt).pow(self.gamma) * bce )
# --------------------
# Dataset
# --------------------
# SeqDataset 교체본: |tgt|<eps 샘플을 시퀀스에서 제외
class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, feats, target_col="tgt"):
        self.seq_len = seq_len
        self.feats = feats
        self.blocks = []   # [(X_np, y_np, valid_mask_np)]
        self.index = []    # [(block_id, start_idx)]
        eps = EPS_BPS / 1e4  # bps → ratio

        for sym, g in df.groupby("symbol", sort=False):
            g = g.dropna(subset=[target_col]).reset_index(drop=True)
            if len(g) < seq_len:
                continue

            X = g[feats].to_numpy(dtype=np.float32, copy=False)
            y = g[target_col].to_numpy(dtype=np.float32, copy=False)

            # 방향 라벨 기준의 노이즈 컷오프: |tgt|<eps → 학습 제외
            valid = (np.abs(y) >= eps)  # bool, 길이=len(g)

            bid = len(self.blocks)
            self.blocks.append((X, y, valid))

            max_start = len(g) - seq_len
            # 시퀀스의 마지막 타깃 위치(e-1)가 유효(valid==True)일 때만 추가
            for s in range(0, max_start):
                e = s + seq_len
                if valid[e-1]:
                    self.index.append((bid, s))

        if not self.index:
            raise ValueError("no sequences after EPS_BPS filtering")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        bid, s = self.index[idx]
        X, y, _valid = self.blocks[bid]
        e = s + self.seq_len
        xi = torch.from_numpy(X[s:e])                  # [L,F]
        yi = torch.tensor(y[e-1], dtype=torch.float32) # scalar
        si = torch.tensor(1.0 if yi.item()>0 else 0.0, dtype=torch.float32)
        return xi, yi, si

# --------------------
# Model
# --------------------
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
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, loss_mode="cls"):
        super().__init__()
        self.loss_mode = loss_mode
        layers = []
        ch_in = in_feat
        for i in range(levels):
            ch_out = hidden
            dilation = 2**i
            layers += [TemporalBlock(ch_in, ch_out, k, dilation, dropout)]
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        if loss_mode == "cls":
            self.head = nn.Linear(hidden, 1)   # logit only
        else:
            self.head = nn.Linear(hidden, 2)   # [reg, logit]

    def forward(self, x):             # x: [B,L,F]
        x = x.transpose(1,2)          # [B,F,L]
        h = self.tcn(x)               # [B,H,L]
        h = h[:, :, -1]               # last step
        o = self.head(h)
        if self.loss_mode == "cls":
            logit = o.squeeze(-1)     # [B]
            return None, logit
        else:
            y_hat = o[:,0]
            logit = o[:,1]
            return y_hat, logit

# --------------------
# Train/Eval
def train_one_epoch(model, dl, opt, loss_mode="cls", bce_lambda=1.0, gamma=2.0):
    class FocalBCEWithLogits(torch.nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__(); self.gamma=gamma
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        def forward(self, logits, targets):
            bce = self.bce(logits, targets)
            p = torch.sigmoid(logits)
            pt = p*targets + (1-p)*(1-targets)
            return (1-pt).pow(self.gamma) * bce

    focal = FocalBCEWithLogits(gamma=gamma)
    huber = torch.nn.SmoothL1Loss(beta=1e-3)
    model.train()
    total = 0.0
    for batch in dl:
        if len(batch) == 4:
            x, y, s, w = batch
        else:
            x, y, s = batch
            w = torch.ones_like(s)
        x, y, s, w = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE), w.to(DEVICE)

        opt.zero_grad()
        y_hat, logit = model(x)
        if loss_mode == "cls":
            loss = (focal(logit, s) * w).mean()
        else:
            loss = huber(y_hat, y) + bce_lambda * (focal(logit, s) * w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
@torch.no_grad()
def evaluate(model, dl, loss_mode="cls", bce_lambda=1.0, gamma=2.0):
    class FocalBCEWithLogits(torch.nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__(); self.gamma=gamma
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        def forward(self, logits, targets):
            bce = self.bce(logits, targets)
            p = torch.sigmoid(logits)
            pt = p*targets + (1-p)*(1-targets)
            return (1-pt).pow(self.gamma) * bce

    focal = FocalBCEWithLogits(gamma=gamma)
    huber = torch.nn.SmoothL1Loss(beta=1e-3)
    model.eval()
    tot, n = 0.0, 0
    logits_all, s_all = [], []

    for batch in dl:
        if len(batch) == 4: x, y, s, w = batch
        else: x, y, s = batch; w = torch.ones_like(s)
        x, y, s, w = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE), w.to(DEVICE)

        y_hat, logit = model(x)
        loss = (focal(logit, s) * w).mean() if loss_mode=="cls" \
               else huber(y_hat,y) + bce_lambda*(focal(logit,s)*w).mean()
        tot += loss.item() * x.size(0); n += x.size(0)

        logits_all.append(logit.detach().cpu())
        s_all.append(s.detach().cpu())

    avg = tot / max(1,n)
    logits = torch.cat(logits_all); labels = torch.cat(s_all)

    # 임계치 스윕: 0.50~0.60
    best_acc, best_thr = 0.0, 0.5
    for thr in torch.linspace(0.50, 0.60, steps=21):
        pred = (torch.sigmoid(logits) > thr).float()
        acc  = (pred == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    return avg, best_acc, best_thr

# --------------------
# Main
# --------------------
def main():
    print(f"[CFG] DATA={DATA_PATH} H={H} L={SEQ_LEN} BATCH={BATCH} EPOCHS={EPOCHS} LR={LR} DEVICE={DEVICE} LOSS_MODE={LOSS_MODE} EPS_BPS={EPS_BPS} MASK={BCE_MASK_SMALL}")

    df = load_df(DATA_PATH)
    df = add_features(df)

    train_df, val_df = split_time(df)

    mu, sd = fit_scaler(train_df, FEATS)
    train_df = apply_scaler(train_df, FEATS, mu, sd)
    val_df   = apply_scaler(val_df,   FEATS, mu, sd)

    tr_ds = SeqDataset(train_df, SEQ_LEN, FEATS, "tgt")
    va_ds = SeqDataset(val_df, SEQ_LEN, FEATS, "tgt")
    # Windows 안전 설정
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False, persistent_workers=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    focal = FocalBCEWithLogits(gamma=2.0)

    model = TCN(in_feat=len(FEATS), hidden=64, levels=5, k=3, dropout=0.3, loss_mode=LOSS_MODE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(reduction="none")  # 가중치를 수동 적용
    huber = None  # joint일 때 내부에서 생성

    best_val = 1e9
    best_acc = 0.0
    best_thr = 0.5
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, tr_dl, opt, loss_mode=LOSS_MODE, bce_lambda=BCE_LAMBDA)
        va_loss, va_acc, va_thr = evaluate(model, va_dl, loss_mode=LOSS_MODE, bce_lambda=BCE_LAMBDA)
        print(f"[E{epoch:02d}] train={tr_loss:.6f}  val={va_loss:.6f}  dir_acc={va_acc * 100:.2f}% thr*={va_thr:.3f}")
        best_thr = va_thr
        # save
        torch.save({"model": model.state_dict(),
                    "cfg": {"H": H, "SEQ_LEN": SEQ_LEN, "FEATS": FEATS, "LOSS_MODE": LOSS_MODE, "BEST_THR": best_thr},
                    "scaler_mu": mu, "scaler_sd": sd}, "tcn_best.pt")

    print(f"[DONE] best_val={best_val:.6f} dir_acc={best_acc*100:.2f}%  saved=tcn_best.pt")
    np.savez("tcn_scaler.npz", mu=mu, sd=sd, feats=np.array(FEATS, dtype=object))

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
