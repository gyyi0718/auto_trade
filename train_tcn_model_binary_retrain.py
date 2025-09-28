# tcn_train_crypto_dir.py (resume 지원)
# -*- coding: utf-8 -*-
import os, math, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
EPS_BPS = float(os.getenv("EPS_BPS", "30.0"))
BCE_MASK_SMALL = int(os.getenv("BCE_MASK_SMALL","1"))

# === Resume 옵션 ===
RESUME      = os.getenv("RESUME","0") in ("1","y","Y","true","True")
CKPT_LATEST = os.getenv("CKPT_LATEST","tcn_best.pt")
CKPT_BEST   = os.getenv("CKPT_BEST","tcn_best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATS = ["ret","rv","mom","vz"]

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    need = {"timestamp","symbol","close","open","high","low","volume","turnover"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"missing columns: {miss}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    if SYM_KEEP: df = df[df["symbol"].isin(SYM_KEEP)].reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        c = g["close"].astype(float).values
        v = g["volume"].astype(float).values
        ret = np.zeros_like(c, dtype=np.float64); ret[1:] = np.log(c[1:]/np.clip(c[:-1],1e-12,None))
        win=60
        rv = pd.Series(ret).rolling(win, min_periods=win//3).std().bfill().values
        ema_win=60; alpha=2/(ema_win+1); ema=np.zeros_like(c); ema[0]=c[0]
        for i in range(1,len(c)): ema[i]=alpha*c[i]+(1-alpha)*ema[i-1]
        mom = (c/np.maximum(ema,1e-12)-1.0)
        v_ma = pd.Series(v).rolling(win, min_periods=win//3).mean().bfill().values
        v_sd = pd.Series(v).rolling(win, min_periods=win//3).std().replace(0, np.nan).bfill().values
        vz  = (v - v_ma) / np.where(v_sd>0, v_sd, 1.0)
        g["ret"]=ret; g["rv"]=rv; g["mom"]=mom; g["vz"]=vz
        out.append(g)
    df2 = pd.concat(out, ignore_index=True)
    df2["tgt"] = df2.groupby("symbol")["close"].transform(lambda s: np.log(s.shift(-H)/s))
    return df2

def fit_scaler(df: pd.DataFrame, cols):
    mu = df[cols].mean().values
    sd = df[cols].std(ddof=0).replace(0, np.nan).fillna(1.0).values if isinstance(df[cols].std(), pd.Series) else df[cols].std(ddof=0).values
    sd = np.where(sd>0, sd, 1.0)
    return mu, sd

def apply_scaler(df: pd.DataFrame, cols, mu, sd):
    df = df.copy(); df[cols] = (df[cols].values - mu) / sd; return df

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
        p = torch.sigmoid(logits); pt = p*targets + (1-p)*(1-targets)
        return (1-pt).pow(self.gamma) * bce

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, feats, target_col="tgt"):
        self.seq_len = seq_len; self.feats = feats
        self.blocks=[]; self.index=[]
        eps = EPS_BPS / 1e4
        for sym, g in df.groupby("symbol", sort=False):
            g = g.dropna(subset=[target_col]).reset_index(drop=True)
            if len(g) < seq_len: continue
            X = g[feats].to_numpy(dtype=np.float32, copy=False)
            y = g[target_col].to_numpy(dtype=np.float32, copy=False)
            valid = (np.abs(y) >= eps)
            bid = len(self.blocks); self.blocks.append((X,y,valid))
            max_start = len(g) - seq_len
            for s in range(0, max_start):
                e = s + seq_len
                if valid[e-1]: self.index.append((bid, s))
        if not self.index: raise ValueError("no sequences after EPS_BPS filtering")
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        bid, s = self.index[idx]; X,y,_v = self.blocks[bid]; e = s+self.seq_len
        xi = torch.from_numpy(X[s:e]); yi = torch.tensor(y[e-1], dtype=torch.float32)
        si = torch.tensor(1.0 if yi.item()>0 else 0.0, dtype=torch.float32)
        return xi, yi, si

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size=chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

def weight_norm_conv(in_c, out_c, k, dilation):
    pad=(k-1)*dilation
    return nn.utils.weight_norm(nn.Conv1d(in_c,out_c,kernel_size=k,padding=pad,dilation=dilation))

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation, dropout):
        super().__init__()
        self.conv1=weight_norm_conv(in_c,out_c,k,dilation); self.chomp1=Chomp1d((k-1)*dilation)
        self.relu1=nn.ReLU(); self.drop1=nn.Dropout(dropout)
        self.conv2=weight_norm_conv(out_c,out_c,k,dilation); self.chomp2=Chomp1d((k-1)*dilation)
        self.relu2=nn.ReLU(); self.drop2=nn.Dropout(dropout)
        self.downsample=nn.Conv1d(in_c,out_c,1) if in_c!=out_c else None
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.conv1(x); out=self.chomp1(out); out=self.relu1(out); out=self.drop1(out)
        out=self.conv2(out); out=self.chomp2(out); out=self.relu2(out); out=self.drop2(out)
        res=x if self.downsample is None else self.downsample(x)
        return self.relu(out+res)

class TCN(nn.Module):
    def __init__(self, in_feat, hidden=128, levels=6, k=3, dropout=0.1, loss_mode="cls"):
        super().__init__(); self.loss_mode=loss_mode
        layers=[]; ch_in=in_feat
        for i in range(levels):
            ch_out=hidden; dilation=2**i
            layers += [TemporalBlock(ch_in,ch_out,k,dilation,dropout)]
            ch_in=ch_out
        self.tcn=nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1 if loss_mode=="cls" else 2)
    def forward(self,x):
        x=x.transpose(1,2); h=self.tcn(x); h=h[:,:, -1]; o=self.head(h)
        if self.loss_mode=="cls":
            return None, o.squeeze(-1)
        else:
            return o[:,0], o[:,1]

def train_one_epoch(model, dl, opt, loss_mode="cls", bce_lambda=1.0, gamma=2.0):
    focal=FocalBCEWithLogits(gamma=gamma); huber=nn.SmoothL1Loss(beta=1e-3)
    model.train(); total=0.0
    for batch in dl:
        x,y,s = batch; x,y,s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
        opt.zero_grad()
        y_hat, logit = model(x)
        if loss_mode=="cls":
            loss = focal(logit, s).mean()
        else:
            loss = huber(y_hat, y) + bce_lambda * focal(logit, s).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total += loss.item()*x.size(0)
    return total/len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, loss_mode="cls", bce_lambda=1.0, gamma=2.0):
    focal=FocalBCEWithLogits(gamma=gamma); huber=nn.SmoothL1Loss(beta=1e-3)
    model.eval(); tot,n=0.0,0; logits_all=[]; s_all=[]
    for batch in dl:
        x,y,s = batch; x,y,s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
        y_hat, logit = model(x)
        loss = focal(logit,s).mean() if loss_mode=="cls" else huber(y_hat,y)+bce_lambda*focal(logit,s).mean()
        tot += loss.item()*x.size(0); n += x.size(0)
        logits_all.append(logit.detach().cpu()); s_all.append(s.detach().cpu())
    avg = tot/max(1,n)
    logits=torch.cat(logits_all); labels=torch.cat(s_all)
    best_acc, best_thr = 0.0, 0.5
    for thr in torch.linspace(0.50,0.60,steps=21):
        pred=(torch.sigmoid(logits)>thr).float()
        acc=(pred==labels).float().mean().item()
        if acc>best_acc: best_acc, best_thr = acc, float(thr)
    return avg, best_acc, best_thr

def save_ckpt(path, epoch, model, opt, mu, sd, best_acc, best_thr):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler_mu": mu,
        "scaler_sd": sd,
        "cfg": {"H":H,"SEQ_LEN":SEQ_LEN,"FEATS":FEATS,"LOSS_MODE":LOSS_MODE,
                "BEST_ACC":best_acc,"BEST_THR":best_thr}
    }, path)

def load_ckpt(path, model, opt):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
    mu, sd = ckpt.get("scaler_mu", None), ckpt.get("scaler_sd", None)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("cfg",{}).get("BEST_ACC", 0.0)
    best_thr = ckpt.get("cfg",{}).get("BEST_THR", 0.5)
    return start_epoch, mu, sd, best_acc, best_thr

def main():
    print(f"[CFG] DATA={DATA_PATH} H={H} L={SEQ_LEN} BATCH={BATCH} EPOCHS={EPOCHS} LR={LR} DEVICE={DEVICE} LOSS_MODE={LOSS_MODE} EPS_BPS={EPS_BPS} MASK={BCE_MASK_SMALL}")

    df = load_df(DATA_PATH)
    df = add_features(df)
    train_df, val_df = split_time(df)

    # 스케일러: resume 시 ckpt의 mu/sd 우선 적용
    mu, sd = None, None
    start_epoch = 1
    best_acc, best_thr = 0.0, 0.5

    model = TCN(in_feat=len(FEATS), hidden=64, levels=5, k=3, dropout=0.3, loss_mode=LOSS_MODE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    if RESUME and os.path.isfile(CKPT_LATEST):
        se, mu_ck, sd_ck, best_acc, best_thr = load_ckpt(CKPT_LATEST, model, opt)
        if mu_ck is not None and sd_ck is not None:
            mu, sd = mu_ck, sd_ck
        start_epoch = max(1, se)
        print(f"[RESUME] from {CKPT_LATEST} -> start_epoch={start_epoch} best_acc={best_acc:.4f} thr={best_thr:.3f}")

    # 스케일러가 없으면 새로 적합
    if mu is None or sd is None:
        mu, sd = fit_scaler(train_df, FEATS)

    train_df = apply_scaler(train_df, FEATS, mu, sd)
    val_df   = apply_scaler(val_df,   FEATS, mu, sd)

    tr_ds = SeqDataset(train_df, SEQ_LEN, FEATS, "tgt")
    va_ds = SeqDataset(val_df, SEQ_LEN, FEATS, "tgt")
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False, persistent_workers=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)

    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loss = train_one_epoch(model, tr_dl, opt, loss_mode=LOSS_MODE, bce_lambda=BCE_LAMBDA)
        va_loss, va_acc, va_thr = evaluate(model, va_dl, loss_mode=LOSS_MODE, bce_lambda=BCE_LAMBDA)
        print(f"[E{epoch:02d}] train={tr_loss:.6f}  val={va_loss:.6f}  dir_acc={va_acc*100:.2f}% thr*={va_thr:.3f}")

        # 최신 저장
        save_ckpt(CKPT_LATEST, epoch, model, opt, mu, sd, va_acc, va_thr)

        # 베스트 갱신 시 저장
        if va_acc > best_acc:
            best_acc, best_thr = va_acc, va_thr
            save_ckpt(CKPT_BEST, epoch, model, opt, mu, sd, best_acc, best_thr)

    print(f"[DONE] best_dir_acc={best_acc*100:.2f}% thr={best_thr:.3f} saved best={CKPT_BEST} latest={CKPT_LATEST}")
    np.savez("tcn_scaler.npz", mu=mu, sd=sd, feats=np.array(FEATS, dtype=object))

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    main()
