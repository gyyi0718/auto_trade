# train_daily_limit_tp_v2.py
# -*- coding: utf-8 -*-
"""
레이블 누수 제거 버전
- 방향 예측 대신 "다음날 수익률" 직접 예측
- 또는 양방향 모두 시뮬레이션
"""
import sys, os, math, json, argparse, warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------- Label V2: 단순 방향 예측 (오버피팅 방지) ----------
def make_labels_simple(df: pd.DataFrame, H: int = 1):
    """
    단순한 방향 예측 (레이블 누수 제거):
    - 오늘(i) 시점에서 예측
    - 실제 진입: 다음날(i+1) open
    - 실제 청산: H일 후(i+H+1) close
    - 상승하면 LONG, 하락하면 SHORT
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    side = np.zeros(len(df), dtype=np.int8)
    returns = np.zeros(len(df), dtype=np.float32)

    for sym, g in df.groupby("symbol", sort=False):
        op = g["open"].to_numpy(float)
        cl = g["close"].to_numpy(float)
        n = len(g)

        for i in range(n - H - 1):  # 다음날 open과 H일 후 close 필요
            # 진입: 다음날 open
            entry_price = op[i + 1]

            # 청산: H일 후 close (진입일 기준)
            exit_price = cl[i + H + 1]

            # 수익률 계산
            future_return = (exit_price / entry_price - 1.0) * 1e4  # bps

            idx = g.index[i]
            side[idx] = 1 if future_return > 0 else 0
            returns[idx] = abs(future_return)

    return (pd.Series(side, index=df.index, name="side"),
            pd.Series(returns, index=df.index, name="expected_bps"))
# ---------- 또는 양방향 시뮬레이션 ----------
def make_labels_both_directions(df: pd.DataFrame, H: int = 1, tp_bps: float = 150):
    """
    양방향 모두 시뮬레이션:
    - LONG 시뮬: 다음날 open → H일 내 TP 도달?
    - SHORT 시뮬: 다음날 open → H일 내 TP 도달?
    - 둘 다 성공 또는 실패 가능 → 확률적 레이블
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 3-class: 0=SHORT_ONLY, 1=BOTH/NEUTRAL, 2=LONG_ONLY
    direction = np.ones(len(df), dtype=np.int8)  # 기본값 1 (NEUTRAL)

    for sym, g in df.groupby("symbol", sort=False):
        op = g["open"].to_numpy(float)
        hi = g["high"].to_numpy(float)
        lo = g["low"].to_numpy(float)
        n = len(g)

        for i in range(n - 1):
            j1 = i + 1
            j2 = min(n, i + H + 1)
            if j2 <= j1: continue

            entry = op[j1]
            tp_long = entry * (1 + tp_bps / 1e4)
            tp_short = entry * (1 - tp_bps / 1e4)

            # H일 내 도달 여부
            long_hit = (hi[j1:j2] >= tp_long).any()
            short_hit = (lo[j1:j2] <= tp_short).any()

            idx = g.index[i]
            if long_hit and not short_hit:
                direction[idx] = 2  # LONG_ONLY
            elif short_hit and not long_hit:
                direction[idx] = 0  # SHORT_ONLY
            else:
                direction[idx] = 1  # BOTH or NEITHER (거래 안함)

    return pd.Series(direction, index=df.index, name="direction")


# ---------- Features (동일) ----------
def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """피처 생성 함수 (수정됨)"""
    g = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    # ===== 변동성 (수정) =====
    for w in (5, 10, 20, 60):
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).fillna(0.0)

    # ===== 모멘텀 (수정) =====
    for w in (5, 10, 20, 60):
        ema = g.groupby("symbol")["close"].transform(
            lambda s: s.ewm(span=w, adjust=False).mean()
        )
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)

    # ===== 거래량 Z-score (수정) =====
    for w in (10, 20, 60):
        mu = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        sd = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    # ===== ATR =====
    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.groupby(g["symbol"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    ).fillna(0.0)

    feats = [
        "ret1", "rv5", "rv10", "rv20", "rv60",
        "mom5", "mom10", "mom20", "mom60",
        "vz10", "vz20", "vz60", "atr14"
    ]

    # NaN 값 처리
    for feat in feats:
        g[feat] = g[feat].fillna(0.0)
        g[feat] = g[feat].replace([np.inf, -np.inf], 0.0)

    return g, feats

# ---------- Dataset (간단한 버전) ----------
class SeqDS(Dataset):
    def __init__(self, df: pd.DataFrame, feats: List[str], seq_len: int,
                 mu: Optional[np.ndarray] = None, sd: Optional[np.ndarray] = None):
        self.df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.feats = feats;
        self.seq = seq_len
        X = self.df[self.feats].to_numpy(np.float32)
        if mu is None or sd is None:
            self.mu = X.mean(0).astype(np.float32)
            self.sd = X.std(0).astype(np.float32);
            self.sd[self.sd == 0] = 1.0
        else:
            self.mu = mu.astype(np.float32);
            self.sd = sd.astype(np.float32)
        self.X = (X - self.mu) / self.sd

        # 단순 레이블 (방향만)
        self.side = self.df["side"].to_numpy(np.int64)

        self.sym_idx = self.df.groupby("symbol").cumcount().to_numpy()
        self.valid = np.where(self.sym_idx >= (self.seq - 1))[0]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        idx = self.valid[i];
        s = idx - self.seq + 1
        x = self.X[s:idx + 1]
        return (
            torch.from_numpy(x),
            torch.tensor(self.side[idx], dtype=torch.long),
        )


# ---------- TCN (동일) ----------
class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c

    def forward(self, x): return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d): import torch.nn.utils as U; pad = (k - 1) * d; return U.weight_norm(
    nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d);
        self.h1 = Chomp1d((k - 1) * d);
        self.r1 = nn.ReLU();
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d);
        self.h2 = Chomp1d((k - 1) * d);
        self.r2 = nn.ReLU();
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None;
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


class TCN_Simple(nn.Module):
    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = [];
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop));
            ch = hidden
        self.tcn = nn.Sequential(*L)
        self.head_side = nn.Linear(hidden, 2)  # Binary classification

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]
        return self.head_side(H)


# ---------- Train ----------
def chrono_split(df: pd.DataFrame, val_ratio: float = 0.15):
    g = df.groupby("symbol", sort=False)
    idx = g.cumcount();
    n = g["date"].transform("size")
    cut = (n * (1.0 - val_ratio)).astype(int)
    return (idx < cut).to_numpy()


def train(df_raw: pd.DataFrame, seq_len=60, horizon=1, batch=256, epochs=50, lr=1e-3, device="cuda"):
    df_feat, feat_cols = make_features(df_raw)

    # 단순 레이블
    side, _ = make_labels_simple(df_feat, H=horizon)
    df_feat["side"] = side
    df_feat = df_feat.dropna(subset=feat_cols + ["side"]).reset_index(drop=True)

    is_tr = chrono_split(df_feat, 0.15)
    tr, va = df_feat[is_tr].reset_index(drop=True), df_feat[~is_tr].reset_index(drop=True)

    X_tr = tr[feat_cols].to_numpy(np.float32)
    mu, sd = X_tr.mean(0).astype(np.float32), X_tr.std(0).astype(np.float32);
    sd[sd == 0] = 1.0

    ds_tr = SeqDS(tr, feat_cols, seq_len, mu, sd)
    ds_va = SeqDS(va, feat_cols, seq_len, mu, sd) if len(va) > 0 else None
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=0) if ds_va else None

    model = TCN_Simple(in_f=len(feat_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_va_acc = -1.0

    for ep in range(1, epochs + 1):
        model.train();
        trn = 0;
        corr = 0;
        n = 0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}", dynamic_ncols=True, disable=not sys.stdout.isatty())
        for X, y_side in pbar:
            X = X.to(device);
            y_side = y_side.to(device)

            logits = model(X)
            loss = ce(logits, y_side)

            opt.zero_grad();
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                corr += (pred == y_side).sum().item()
                n += y_side.numel()
                trn += loss.item() * y_side.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{corr / max(n, 1):.3f}")
        pbar.close()

        # validation
        va_acc = 0;
        va_n = 0;
        va_loss = 0.0
        if dl_va:
            model.eval()
            with torch.no_grad():
                for X, y_side in dl_va:
                    X = X.to(device);
                    y_side = y_side.to(device)
                    logits = model(X)
                    loss = ce(logits, y_side)
                    va_loss += loss.item() * y_side.size(0)
                    va_acc += (logits.argmax(1) == y_side).sum().item()
                    va_n += y_side.numel()

        msg = f"[{ep:03d}] tr_loss={trn / max(n, 1):.4f} tr_acc={corr / max(n, 1):.3f} | "
        if dl_va:
            msg += f"va_loss={va_loss / max(va_n, 1):.4f} va_acc={va_acc / max(va_n, 1):.3f}"
        else:
            msg += "va_loss=NA va_acc=NA"
        print(msg)

        # Save checkpoint
        ckpt = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "feat_cols": feat_cols,
            "meta": {"seq_len": seq_len, "horizon": horizon},
            "scaler_mu": mu.copy(),
            "scaler_sd": sd.copy(),
        }
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.out_dir, f"daily_simple_ep{ep:03d}.ckpt"))

        if dl_va:
            va_acc_val = va_acc / max(va_n, 1)
            if va_acc_val > best_va_acc:
                best_va_acc = va_acc_val
                torch.save(ckpt, os.path.join(args.out_dir, "daily_simple_best.ckpt"))
                print(f"[BEST] epoch {ep} | va_acc={va_acc_val:.4f}")

    return ckpt, feat_cols, {"seq_len": seq_len, "horizon": horizon, "scaler_mu": mu, "scaler_sd": sd}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="./models_daily_v2")
    return ap.parse_args()


def main(args):
    if args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    if not is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        if is_datetime64tz_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_convert("UTC")
        else:
            df["date"] = df["date"].dt.tz_localize("UTC")

    need = {"date", "symbol", "open", "high", "low", "close", "volume"}
    assert need.issubset(df.columns), f"missing cols: {need - set(df.columns)}"
    df = df.dropna(subset=list(need)).sort_values(["symbol", "date"]).reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_state, feats, meta = train(df, seq_len=args.seq_len, horizon=args.horizon,
                                    batch=args.batch, epochs=args.epochs, lr=args.lr, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    meta_json = {
        "feat_cols": feats,
        "seq_len": meta["seq_len"],
        "horizon": meta["horizon"],
        "scaler_mu": meta["scaler_mu"].tolist(),
        "scaler_sd": meta["scaler_sd"].tolist()
    }
    with open(os.path.join(args.out_dir, "daily_simple_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)
    print("[DONE] saved to", args.out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)

    #python train_tcn_daily.py --data daily_ohlcv.csv