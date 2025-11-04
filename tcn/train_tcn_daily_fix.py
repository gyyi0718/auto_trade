# train_tcn_daily_fixed.py
# -*- coding: utf-8 -*-
"""
개선 사항:
1. Rolling Window Normalization (시장 변화 적응)
2. Confidence Score 추가 (애매한 신호 필터링)
3. Robust Feature Engineering
4. 이중 정규화 제거
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


# ---------- Label (동일) ----------
def make_labels_simple(df: pd.DataFrame, H: int = 1):
    """단순한 방향 예측"""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    side = np.zeros(len(df), dtype=np.int8)
    returns = np.zeros(len(df), dtype=np.float32)

    for sym, g in df.groupby("symbol", sort=False):
        op = g["open"].to_numpy(float)
        cl = g["close"].to_numpy(float)
        n = len(g)

        for i in range(n - H - 1):
            entry_price = op[i + 1]
            exit_price = cl[i + H + 1]
            future_return = (exit_price / entry_price - 1.0) * 1e4

            idx = g.index[i]
            side[idx] = 1 if future_return > 0 else 0
            returns[idx] = abs(future_return)

    return (pd.Series(side, index=df.index, name="side"),
            pd.Series(returns, index=df.index, name="expected_bps"))


# ---------- Features (개선됨) ----------
def make_features_robust(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    더 Robust한 피처 생성
    - 이미 정규화된 피처들만 사용 (rolling 기반)
    - 절대값 대신 상대값 위주
    """
    g = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    # ===== 1. 수익률 (이미 정규화됨) =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    # ===== 2. 변동성 (Rolling Std) =====
    for w in (5, 10, 20):
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).fillna(0.0)

    # ===== 3. 모멘텀 (상대값, 이미 정규화) =====
    for w in (5, 10, 20):
        ema = g.groupby("symbol")["close"].transform(
            lambda s: s.ewm(span=w, adjust=False).mean()
        )
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)

    # ===== 4. 거래량 Z-score (Rolling, 이미 정규화) =====
    for w in (10, 20):
        mu = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        sd = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)
        # Clip to prevent extreme values
        g[f"vz{w}"] = g[f"vz{w}"].clip(-5, 5)

    # ===== 5. ATR (상대값으로 변환) =====
    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.groupby(g["symbol"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    ).fillna(0.0)
    # ATR을 가격 대비 비율로 (더 robust)
    g["atr_ratio"] = (atr / g["close"]).fillna(0.0).clip(0, 0.5)

    # ===== 6. 추가: 단기 모멘텀 변화율 =====
    g["mom_accel"] = g.groupby("symbol")["mom5"].diff().fillna(0.0)

    feats = [
        "ret1", "rv5", "rv10", "rv20",
        "mom5", "mom10", "mom20", "mom_accel",
        "vz10", "vz20", "atr_ratio"
    ]

    # NaN 및 Inf 처리
    for feat in feats:
        g[feat] = g[feat].fillna(0.0)
        g[feat] = g[feat].replace([np.inf, -np.inf], 0.0)
        # 추가 보호: 극단값 클리핑
        if feat not in ["ret1"]:  # ret1은 클리핑 안함
            q99 = g[feat].quantile(0.99)
            q01 = g[feat].quantile(0.01)
            g[feat] = g[feat].clip(q01, q99)

    return g, feats


# ---------- Dataset (Rolling Normalization) ----------
class SeqDS_Rolling(Dataset):
    """
    Rolling Window Normalization
    - 고정된 scaler 대신 최근 데이터 기준으로 정규화
    """

    def __init__(self, df: pd.DataFrame, feats: List[str], seq_len: int,
                 normalize_window: int = 60):
        self.df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.feats = feats
        self.seq = seq_len
        self.norm_window = normalize_window

        # Feature 데이터 (정규화 안된 상태)
        self.X_raw = self.df[self.feats].to_numpy(np.float32)

        # 레이블
        self.side = self.df["side"].to_numpy(np.int64)

        # 심볼별 인덱스
        self.sym_idx = self.df.groupby("symbol").cumcount().to_numpy()

        # 유효한 인덱스 (seq_len + norm_window 확보 필요)
        min_required = max(self.seq, self.norm_window) - 1
        self.valid = np.where(self.sym_idx >= min_required)[0]

        print(f"Dataset initialized: {len(self.valid)} valid samples")
        print(f"Normalization window: {normalize_window} days")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        idx = self.valid[i]

        # Sequence 범위
        s = idx - self.seq + 1

        # Rolling normalization을 위한 window 범위
        # norm_window 기간의 통계로 정규화
        norm_start = max(0, idx - self.norm_window + 1)
        norm_end = idx + 1

        # 정규화 통계 계산 (최근 norm_window 기간)
        X_norm_data = self.X_raw[norm_start:norm_end]
        mu = X_norm_data.mean(axis=0, keepdims=True).astype(np.float32)
        sd = X_norm_data.std(axis=0, keepdims=True).astype(np.float32)
        sd[sd < 1e-6] = 1.0  # Zero division 방지

        # Sequence 데이터 정규화
        X_seq = self.X_raw[s:idx + 1]
        X_normalized = (X_seq - mu) / sd

        # Inf/NaN 체크
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=5.0, neginf=-5.0)

        return (
            torch.from_numpy(X_normalized),
            torch.tensor(self.side[idx], dtype=torch.long),
        )


# ---------- TCN with Confidence ----------
class TCN_Confidence(nn.Module):
    """
    Confidence Score를 함께 출력하는 TCN
    """

    def __init__(self, in_f, hidden=128, levels=6, k=3, drop=0.2):
        super().__init__()
        L = []
        ch = in_f
        for i in range(levels):
            L.append(Block(ch, hidden, k, 2 ** i, drop))
            ch = hidden
        self.tcn = nn.Sequential(*L)

        # 방향 예측
        self.head_side = nn.Linear(hidden, 2)

        # Confidence 예측 (0~1, 얼마나 확신하는가)
        self.head_conf = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0~1 범위
        )

    def forward(self, X):
        X = X.transpose(1, 2)
        H = self.tcn(X)[:, :, -1]

        side_logits = self.head_side(H)
        confidence = self.head_conf(H).squeeze(-1)

        return side_logits, confidence


# ---------- TCN Blocks (동일) ----------
class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x


def wconv(i, o, k, d):
    import torch.nn.utils as U
    pad = (k - 1) * d
    return U.weight_norm(nn.Conv1d(i, o, k, padding=pad, dilation=d))


class Block(nn.Module):
    def __init__(self, i, o, k, d, drop):
        super().__init__()
        self.c1 = wconv(i, o, k, d)
        self.h1 = Chomp1d((k - 1) * d)
        self.r1 = nn.ReLU()
        self.dr1 = nn.Dropout(drop)
        self.c2 = wconv(o, o, k, d)
        self.h2 = Chomp1d((k - 1) * d)
        self.r2 = nn.ReLU()
        self.dr2 = nn.Dropout(drop)
        self.ds = nn.Conv1d(i, o, 1) if i != o else None
        self.r = nn.ReLU()

    def forward(self, x):
        y = self.dr1(self.r1(self.h1(self.c1(x))))
        y = self.dr2(self.r2(self.h2(self.c2(y))))
        res = x if self.ds is None else self.ds(x)
        return self.r(y + res)


# ---------- Train ----------
def chrono_split(df: pd.DataFrame, val_ratio: float = 0.15):
    g = df.groupby("symbol", sort=False)
    idx = g.cumcount()
    n = g["date"].transform("size")
    cut = (n * (1.0 - val_ratio)).astype(int)
    return (idx < cut).to_numpy()


def train(df_raw: pd.DataFrame, seq_len=60, horizon=1, batch=256, epochs=50,
          lr=1e-3, device="cuda", normalize_window=60):
    print("Generating features...")
    df_feat, feat_cols = make_features_robust(df_raw)

    print("Creating labels...")
    side, expected_bps = make_labels_simple(df_feat, H=horizon)
    df_feat["side"] = side
    df_feat["expected_bps"] = expected_bps
    df_feat = df_feat.dropna(subset=feat_cols + ["side"]).reset_index(drop=True)

    print(f"Label distribution: SHORT={(side == 0).sum()}, LONG={(side == 1).sum()}")

    is_tr = chrono_split(df_feat, 0.15)
    tr = df_feat[is_tr].reset_index(drop=True)
    va = df_feat[~is_tr].reset_index(drop=True)

    # Rolling normalization dataset
    ds_tr = SeqDS_Rolling(tr, feat_cols, seq_len, normalize_window)
    ds_va = SeqDS_Rolling(va, feat_cols, seq_len, normalize_window) if len(va) > 0 else None

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=0) if ds_va else None

    model = TCN_Confidence(in_f=len(feat_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()  # Confidence loss

    best_va_acc = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        trn_loss = 0
        trn_corr = 0
        trn_n = 0

        pbar = tqdm(dl_tr, desc=f"Epoch {ep}", dynamic_ncols=True,
                    disable=not sys.stdout.isatty())

        for X, y_side in pbar:
            X = X.to(device)
            y_side = y_side.to(device)

            side_logits, confidence = model(X)

            # Loss 1: Classification
            loss_cls = ce(side_logits, y_side)

            # Loss 2: Confidence (정답 맞추면 높게, 틀리면 낮게)
            pred_side = side_logits.argmax(1)
            correct = (pred_side == y_side).float()
            loss_conf = mse(confidence, correct)

            # Total loss
            loss = loss_cls + 0.3 * loss_conf

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                trn_corr += (pred_side == y_side).sum().item()
                trn_n += y_side.numel()
                trn_loss += loss.item() * y_side.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{trn_corr / max(trn_n, 1):.3f}")

        pbar.close()

        # Validation
        va_acc = 0
        va_n = 0
        va_loss = 0.0
        va_conf_avg = 0.0

        if dl_va:
            model.eval()
            with torch.no_grad():
                for X, y_side in dl_va:
                    X = X.to(device)
                    y_side = y_side.to(device)

                    side_logits, confidence = model(X)
                    pred_side = side_logits.argmax(1)
                    correct = (pred_side == y_side).float()

                    loss_cls = ce(side_logits, y_side)
                    loss_conf = mse(confidence, correct)
                    loss = loss_cls + 0.3 * loss_conf

                    va_loss += loss.item() * y_side.size(0)
                    va_acc += (pred_side == y_side).sum().item()
                    va_conf_avg += confidence.sum().item()
                    va_n += y_side.numel()

        msg = f"[{ep:03d}] tr_loss={trn_loss / max(trn_n, 1):.4f} tr_acc={trn_corr / max(trn_n, 1):.3f} | "
        if dl_va:
            msg += f"va_loss={va_loss / max(va_n, 1):.4f} va_acc={va_acc / max(va_n, 1):.3f} "
            msg += f"va_conf={va_conf_avg / max(va_n, 1):.3f}"
        else:
            msg += "va_loss=NA va_acc=NA"
        print(msg)

        # Save checkpoint
        ckpt = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "feat_cols": feat_cols,
            "meta": {
                "seq_len": seq_len,
                "horizon": horizon,
                "normalize_window": normalize_window
            },
        }
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.out_dir, f"daily_fixed_ep{ep:03d}.ckpt"))

        if dl_va:
            va_acc_val = va_acc / max(va_n, 1)
            if va_acc_val > best_va_acc:
                best_va_acc = va_acc_val
                torch.save(ckpt, os.path.join(args.out_dir, "daily_fixed_best.ckpt"))
                print(f"[BEST] epoch {ep} | va_acc={va_acc_val:.4f}")

    return ckpt, feat_cols, {"seq_len": seq_len, "horizon": horizon,
                             "normalize_window": normalize_window}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--normalize_window", type=int, default=60,
                    help="Rolling window for normalization (days)")
    ap.add_argument("--out_dir", type=str, default="./models_daily_fixed")
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
    print(f"Using device: {device}")

    best_state, feats, meta = train(
        df,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        normalize_window=args.normalize_window
    )

    os.makedirs(args.out_dir, exist_ok=True)
    meta_json = {
        "feat_cols": feats,
        "seq_len": meta["seq_len"],
        "horizon": meta["horizon"],
        "normalize_window": meta["normalize_window"]
    }
    with open(os.path.join(args.out_dir, "daily_fixed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)
    print("[DONE] saved to", args.out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)