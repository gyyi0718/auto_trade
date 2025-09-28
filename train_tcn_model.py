# tcn_train_crypto.py
# -*- coding: utf-8 -*-
"""
TCN for multi-symbol 1m futures
- 입력: bybit_futures_1m.csv/parquet (columns: timestamp,symbol,open,high,low,close,volume,turnover)
- 타깃: H분 후 로그수익률 y = log(close[t+H]/close[t])
- 손실: Huber(회귀) + BCE(부호)
- 저장: 모델 .pt, 표준화 파라미터 .npz
Env:
  DATA_PATH=bybit_futures_1m.parquet|csv
  H=3                  # 예측 지평(분)
  SEQ_LEN=240
  BATCH=512
  EPOCHS=30
  LR=1e-3
  SYM_KEEP=ETHUSDT,BTCUSDT,...   # 선택 심볼(미지정이면 전부)
  TRAIN_DAYS=150  VAL_DAYS=30    # 시계열 분할
"""

import os, math, gc, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------
# Config
# --------------------
DATA_PATH = os.getenv("DATA_PATH", "bybit_futures_1m.parquet")
H         = int(os.getenv("H", "3"))
SEQ_LEN   = int(os.getenv("SEQ_LEN", "60"))
BATCH     = int(os.getenv("BATCH", "512"))
EPOCHS    = int(os.getenv("EPOCHS", "30"))
LR        = float(os.getenv("LR", "1e-3"))
TRAIN_DAYS= int(os.getenv("TRAIN_DAYS", "30"))
VAL_DAYS  = int(os.getenv("VAL_DAYS", "30"))
SYM_KEEP  = [s for s in os.getenv("SYM_KEEP","").split(",") if s.strip()]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# I/O
# --------------------
def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # 기본 컬럼 확인
    need = {"timestamp","symbol","close","open","high","low","volume","turnover"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")
    # 정렬
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

        # 1) log return
        ret = np.zeros_like(c, dtype=np.float64)
        ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))

        # 2) rolling vol (stdev of returns, 60m)
        win = 60
        rv = pd.Series(ret).rolling(win, min_periods=win // 3).std().bfill().values

        # 3) momentum (close/EMA)
        ema_win = 60
        alpha = 2/(ema_win+1)
        ema = np.zeros_like(c, dtype=np.float64)
        ema[0] = c[0]
        for i in range(1,len(c)):
            ema[i] = alpha*c[i] + (1-alpha)*ema[i-1]
        mom = (c/np.maximum(ema,1e-12) - 1.0)

        # 4) volume z-score
        # rv, v_ma, v_sd 계산부
        v_ma = pd.Series(v).rolling(win, min_periods=win // 3).mean().bfill().values
        v_sd = pd.Series(v).rolling(win, min_periods=win // 3).std().replace(0, np.nan).bfill().values
        vz  = (v - v_ma) / np.where(v_sd>0, v_sd, 1.0)

        g["ret"] = ret
        g["rv"]  = rv
        g["mom"] = mom
        g["vz"]  = vz
        out.append(g)
    df2 = pd.concat(out, ignore_index=True)
    # target H-step return
    df2["tgt"] = df2.groupby("symbol")["close"].transform(lambda s: np.log(s.shift(-H) / s))
    return df2

# --------------------
# Scaling
# --------------------
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
# Dataset
# --------------------
FEATS = ["ret","rv","mom","vz"]
def split_time(df: pd.DataFrame):
    # 최근(VAL_DAYS) 검증, 그 이전(TRAIN_DAYS) 학습. 나머지는 버림.
    tmax = df["timestamp"].max()
    val_start = tmax - pd.Timedelta(days=VAL_DAYS)
    train_start = val_start - pd.Timedelta(days=TRAIN_DAYS)
    train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < val_start)].copy()
    val   = df[(df["timestamp"] >= val_start)].copy()
    return train, val

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, feats, target_col="tgt"):
        self.seq_len = seq_len
        self.feats = feats
        self.blocks = []   # [(X_np, y_np)]
        self.index = []    # [(block_id, start_idx)]

        for sym, g in df.groupby("symbol", sort=False):
            g = g.dropna(subset=[target_col]).reset_index(drop=True)
            if len(g) < seq_len:
                continue
            X = g[feats].to_numpy(dtype=np.float32, copy=False)
            y = g[target_col].to_numpy(dtype=np.float32, copy=False)
            bid = len(self.blocks)
            self.blocks.append((X, y))
            # 시퀀스 시작 위치만 기록
            max_start = len(g) - seq_len
            for s in range(0, max_start):
                self.index.append((bid, s))
        if not self.index:
            raise ValueError("no sequences")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        bid, s = self.index[idx]
        X, y = self.blocks[bid]
        e = s + self.seq_len
        xi = torch.from_numpy(X[s:e])           # [L,F]
        yi = torch.tensor(y[e-1], dtype=torch.float32)  # 마지막 시점 타깃
        sign = torch.tensor(1.0 if yi.item()>0 else 0.0, dtype=torch.float32)
        return xi, yi, sign

# --------------------
# TCN Model
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
    def __init__(self, in_feat, hidden=64, levels=5, k=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_feat
        for i in range(levels):
            ch_out = hidden
            dilation = 2**i
            layers += [TemporalBlock(ch_in, ch_out, k, dilation, dropout)]
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)   # out[0]=reg(y), out[1]=logit(sign)
    def forward(self, x):             # x: [B,L,F]
        x = x.transpose(1,2)          # [B,F,L]
        h = self.tcn(x)               # [B,H,L]
        h = h[:, :, -1]               # last step
        o = self.head(h)              # [B,2]
        y_hat = o[:,0]
        logit = o[:,1]
        return y_hat, logit

# --------------------
# Train/Eval
# --------------------
def train_one_epoch(model, dl, opt, huber, bce, w_bce=0.3):
    model.train()
    total = 0.0
    for x, y, s in dl:
        x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
        opt.zero_grad()
        y_hat, logit = model(x)
        loss = huber(y_hat, y) + w_bce * bce(logit, s)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, huber, bce, w_bce=0.3):
    model.eval()
    tot, n = 0.0, 0
    hit, count = 0, 0
    for x, y, s in dl:
        x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
        y_hat, logit = model(x)
        loss = huber(y_hat, y) + w_bce * bce(logit, s)
        tot += loss.item() * x.size(0)
        n += x.size(0)
        pred = (logit.sigmoid() > 0.5).float()
        hit += (pred == s).sum().item()
        count += s.numel()
    avg = tot / max(1,n)
    acc = hit / max(1,count)
    return avg, acc

# --------------------
# Main
# --------------------
def main():
    print(f"[CFG] DATA={DATA_PATH} H={H} L={SEQ_LEN} BATCH={BATCH} EPOCHS={EPOCHS} LR={LR} DEVICE={DEVICE}")

    df = load_df(DATA_PATH)
    df = add_features(df)

    # 시간 분할
    train_df, val_df = split_time(df)

    # 스케일러(훈련 구간 기준)
    mu, sd = fit_scaler(train_df, FEATS)
    train_df = apply_scaler(train_df, FEATS, mu, sd)
    val_df   = apply_scaler(val_df,   FEATS, mu, sd)

    # 데이터셋
    tr_ds = SeqDataset(train_df, SEQ_LEN, FEATS, "tgt")
    va_ds = SeqDataset(val_df,   SEQ_LEN, FEATS, "tgt")
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    # 모델
    model = TCN(in_feat=len(FEATS), hidden=128, levels=6, k=3, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    huber = nn.SmoothL1Loss(beta=1e-3)  # 작은 오차 중시
    bce   = nn.BCEWithLogitsLoss()

    best_val = 1e9
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, tr_dl, opt, huber, bce, w_bce=0.3)
        va_loss, va_acc = evaluate(model, va_dl, huber, bce, w_bce=0.3)
        print(f"[E{epoch:02d}] train={tr_loss:.6f}  val={va_loss:.6f}  dir_acc={va_acc*100:.2f}%")
        if va_loss < best_val:
            best_val = va_loss
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "cfg": {"H":H,"SEQ_LEN":SEQ_LEN,"FEATS":FEATS},
                        "scaler_mu": mu, "scaler_sd": sd}, "tcn_best.pt")
    print(f"[DONE] best_val={best_val:.6f} dir_acc={best_acc*100:.2f}%  saved=tcn_best.pt")
    # 스케일러 따로 저장(선택)
    np.savez("tcn_scaler.npz", mu=mu, sd=sd, feats=np.array(FEATS, dtype=object))

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()