#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchTST 학습 스크립트 - 기존 TCN 코드와 호환

사용법:
    python train_patchtst.py --data coin_5min.parquet --epochs 50 --seq_len 72

주요 개선사항:
    1. 현실적 레이블링 (슬리피지 포함)
    2. Feature selection (자동)
    3. PatchTST 모델 사용
"""

import sys, os, json, argparse, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==================== 현실적 레이블링 ====================
def make_realistic_labels(df: pd.DataFrame, horizon_candles=72,
                          entry_slippage_bps=10, exit_slippage_bps=10,
                          min_profit_bps=150):
    """슬리피지를 고려한 현실적 레이블 생성"""
    print(f"\n[LABEL] 현실적 레이블 생성")
    print(f"  진입 슬리피지: {entry_slippage_bps / 100:.2f}%")
    print(f"  청산 슬리피지: {exit_slippage_bps / 100:.2f}%")
    print(f"  최소 수익: {min_profit_bps / 100:.2f}%")

    g = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    labels = []

    for sym, gi in g.groupby("symbol", sort=False):
        gi = gi.reset_index(drop=True)

        for i in range(len(gi) - horizon_candles):
            future_data = gi.iloc[i + 1:i + 1 + horizon_candles]
            if len(future_data) < horizon_candles:
                continue

            next_open = float(future_data.iloc[0]["open"])
            long_entry = next_open * (1 + entry_slippage_bps / 1e4)
            short_entry = next_open * (1 - entry_slippage_bps / 1e4)

            high = float(future_data["high"].max())
            low = float(future_data["low"].min())

            long_exit = high * (1 - exit_slippage_bps / 1e4)
            short_exit = low * (1 + exit_slippage_bps / 1e4)

            long_bps = (long_exit / long_entry - 1.0) * 1e4
            short_bps = (short_entry / short_exit - 1.0) * 1e4

            if max(long_bps, short_bps) < min_profit_bps:
                continue

            labels.append({
                "symbol": sym,
                "pred_time": future_data.iloc[0]["date"],
                "side": 1 if long_bps >= short_bps else 0,
                "expected_bps": float(max(long_bps, short_bps)),
                "confidence": abs(long_bps - short_bps)
            })

    df_labels = pd.DataFrame(labels)
    print(f"  ✅ {len(df_labels):,} labels 생성")
    return df_labels


# ==================== Feature Engineering ====================
def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Feature 생성"""
    print("\n[FEATURE] Feature 생성 중...")
    g = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    # 기본
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    # 변동성
    for w in [8, 20, 40]:
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).fillna(0.0)

    # 모멘텀
    for w in [8, 20, 40]:
        ema = g.groupby("symbol")["close"].transform(
            lambda s: s.ewm(span=w, adjust=False).mean()
        )
        g[f"mom{w}"] = (g["close"] / ema - 1.0).fillna(0.0)

    # 볼륨
    for w in [20, 40]:
        mu = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        sd = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0)

    g["vol_spike"] = g.groupby("symbol")["volume"].transform(
        lambda s: s / s.shift(1) - 1.0
    ).fillna(0.0)

    # ATR
    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    g["atr14"] = tr.groupby(g["symbol"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    ).fillna(0.0)

    # 캔들
    g["hl_spread"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0)
    g["close_pos"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5)
    g["body_size"] = ((g["close"] - g["open"]).abs() / g["open"]).fillna(0.0)

    # 수익률
    for w in [2, 4, 8, 12]:
        g[f"ret{w}"] = g.groupby("symbol")["logc"].diff(w).fillna(0.0)

    # RSI
    delta = g.groupby("symbol")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(g["symbol"]).transform(
        lambda x: x.rolling(14, min_periods=7).mean()
    )
    avg_loss = loss.groupby(g["symbol"]).transform(
        lambda x: x.rolling(14, min_periods=7).mean()
    )
    rs = avg_gain / (avg_loss + 1e-10)
    g["rsi14"] = (100 - (100 / (1 + rs)) - 50) / 50
    g["rsi14"] = g["rsi14"].fillna(0.0)

    # MACD
    ema12 = g.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    ema26 = g.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0)

    # 시간
    hod = g["date"].dt.hour
    g["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    dow = g["date"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    feats = [
        "ret1", "ret2", "ret4", "ret8", "ret12",
        "rv8", "rv20", "rv40",
        "mom8", "mom20", "mom40",
        "vz20", "vz40", "vol_spike",
        "atr14", "hl_spread", "close_pos", "body_size",
        "rsi14", "macd",
        "hod_sin", "hod_cos", "dow_sin", "dow_cos"
    ]

    # 정리
    for feat in feats:
        if feat not in ['hod_sin', 'hod_cos', 'dow_sin', 'dow_cos']:
            q01, q99 = g[feat].quantile([0.01, 0.99])
            g[feat] = g[feat].clip(q01, q99)
        g[feat] = g[feat].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    print(f"  ✅ {len(feats)}개 feature 생성")
    return g, feats


# ==================== Feature Selection ====================
def select_features_rf(df_feat, labels, feat_cols, top_k=20):
    """Random Forest로 중요 feature 선택"""
    print(f"\n[SELECTION] {len(feat_cols)} → {top_k} features")

    from sklearn.ensemble import RandomForestClassifier

    # 샘플링
    n_samples = min(30000, len(labels))
    sample_idx = np.random.choice(len(labels), n_samples, replace=False)
    labels_sample = labels.iloc[sample_idx].reset_index(drop=True)

    X_list, y_list = [], []
    for _, row in labels_sample.iterrows():
        sym_data = df_feat[
            (df_feat["symbol"] == row["symbol"]) &
            (df_feat["date"] < pd.to_datetime(row["pred_time"], utc=True))
            ]
        if len(sym_data) < 20:
            continue

        # 최근 20개 평균
        X_list.append(sym_data.iloc[-20:][feat_cols].mean().values)
        y_list.append(row["side"])

    if len(X_list) < 100:
        print("  ⚠️ 샘플 부족, 모든 feature 사용")
        return feat_cols

    X = np.array(X_list)
    y = np.array(y_list)

    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-top_k:]
    selected = [feat_cols[i] for i in top_indices]

    print(f"  ✅ Top {top_k}: {selected[-5:]}")
    return selected


# ==================== Dataset ====================
class PatchDataset(Dataset):
    def __init__(self, feat_df, labels, feats, seq_len, mu=None, sd=None):
        self.df = feat_df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.feats = feats
        self.seq = seq_len

        X = self.df[self.feats].to_numpy(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if mu is None:
            self.mu = X.mean(0).astype(np.float32)
            self.sd = X.std(0).astype(np.float32)
            self.sd[self.sd < 1e-8] = 1.0
        else:
            self.mu, self.sd = mu, sd

        self.X = np.clip((X - self.mu) / self.sd, -10, 10)

        # 인덱스 생성
        sym_idx = {s: self.df[self.df['symbol'] == s].index.values
                   for s in self.df['symbol'].unique()}
        sym_dates = {s: self.df[self.df['symbol'] == s]['date'].values.astype('datetime64[ns]')
                     for s in self.df['symbol'].unique()}

        samples = []
        for _, row in labels.iterrows():
            sym = row["symbol"]
            if sym not in sym_dates:
                continue

            pred_time = np.datetime64(pd.to_datetime(row["pred_time"], utc=True))
            pos = np.searchsorted(sym_dates[sym], pred_time)

            if pos < self.seq:
                continue

            idx = sym_idx[sym]
            samples.append({
                "start": int(idx[pos - self.seq]),
                "end": int(idx[pos - 1]),
                "side": int(row["side"])
            })

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        indices = range(s["start"], s["end"] + 1)
        x = self.X[list(indices)[-self.seq:]]

        if len(x) < self.seq:
            pad = np.zeros((self.seq - len(x), len(self.feats)), dtype=np.float32)
            x = np.vstack([pad, x])

        return torch.from_numpy(x), torch.tensor(s["side"], dtype=torch.long)


# ==================== PatchTST 모델 ====================
class PatchTST(nn.Module):
    def __init__(self, n_features, seq_len, patch_size=12,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(n_features * patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x):
        # x: [B, seq_len, features]
        B = x.size(0)

        # Reshape to patches: [B, n_patches, patch_size * features]
        x = x.reshape(B, self.n_patches, -1)

        # Embed
        x = self.patch_embed(x) + self.pos_embed

        # Transform
        x = self.transformer(x)

        # Pool and classify
        x = x.mean(dim=1)
        return self.classifier(x)


# ==================== 학습 ====================
def train_model(df_raw, args):
    """PatchTST 학습"""
    print("=" * 80)
    print("PatchTST 학습 시작")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Data: {len(df_raw):,} candles, {df_raw['symbol'].nunique()} symbols")
    print(f"Date range: {df_raw['date'].min()} ~ {df_raw['date'].max()}")

    # 1. Feature 생성
    df_feat, all_feats = make_features(df_raw)

    # 2. 레이블 생성
    labels = make_realistic_labels(
        df_raw,
        horizon_candles=args.horizon,
        entry_slippage_bps=args.slippage,
        exit_slippage_bps=args.slippage,
        min_profit_bps=args.min_bps
    )

    if len(labels) < 1000:
        print("\n❌ 레이블이 너무 적습니다. min_bps를 낮추거나 데이터를 늘리세요.")
        return

    # 샘플링
    if args.sample_ratio < 1.0:
        labels = labels.sample(frac=args.sample_ratio, random_state=42)
        print(f"\n[SAMPLE] {len(labels):,} samples ({args.sample_ratio * 100:.0f}%)")

    # 레이블 분포
    print(f"\n[LABEL DIST]")
    print(f"  LONG:  {(labels['side'] == 1).sum():>6} ({(labels['side'] == 1).mean() * 100:.1f}%)")
    print(f"  SHORT: {(labels['side'] == 0).sum():>6} ({(labels['side'] == 0).mean() * 100:.1f}%)")
    print(f"  Avg BPS: {labels['expected_bps'].mean():.1f}")

    # 3. Feature selection
    if args.feature_selection:
        selected_feats = select_features_rf(df_feat, labels, all_feats, top_k=args.top_k)
    else:
        selected_feats = all_feats
        print(f"\n[FEATURE] {len(selected_feats)}개 feature 사용 (selection 비활성화)")

    # 4. Train/Val split
    split_idx = int(len(labels) * 0.8)
    labels = labels.sort_values(["symbol", "pred_time"]).reset_index(drop=True)
    labels_train = labels.iloc[:split_idx]
    labels_val = labels.iloc[split_idx:]

    print(f"\n[SPLIT]")
    print(f"  Train: {len(labels_train):,}")
    print(f"  Val:   {len(labels_val):,}")

    # 5. Dataset
    ds_train = PatchDataset(df_feat, labels_train, selected_feats, args.seq_len)
    ds_val = PatchDataset(df_feat, labels_val, selected_feats, args.seq_len,
                          ds_train.mu, ds_train.sd)

    print(f"  Train samples: {len(ds_train):,}")
    print(f"  Val samples:   {len(ds_val):,}")

    if len(ds_train) == 0:
        print("\n❌ Train 샘플이 없습니다. seq_len을 줄이거나 데이터를 늘리세요.")
        return

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=2)

    # 6. 모델
    model = PatchTST(
        n_features=len(selected_feats),
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] PatchTST")
    print(f"  Parameters: {n_params:,}")
    print(f"  Patch size: {args.patch_size} ({args.patch_size * 5} min)")
    print(f"  d_model: {args.d_model}")
    print(f"  Layers: {args.n_layers}")

    # 7. Loss & Optimizer
    # 클래스 가중치
    n_long = (labels_train['side'] == 1).sum()
    n_short = (labels_train['side'] == 0).sum()
    weight_long = len(labels_train) / (2 * n_long)
    weight_short = len(labels_train) / (2 * n_short)
    class_weights = torch.tensor([weight_short, weight_long], dtype=torch.float32).to(device)

    print(f"\n[LOSS] CrossEntropy with weights")
    print(f"  SHORT: {weight_short:.2f}")
    print(f"  LONG:  {weight_long:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 8. 학습
    print(f"\n[TRAIN] Starting...")
    best_acc = 0
    best_bal_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)

            pbar.set_postfix({
                'loss': f'{train_loss / train_total:.4f}',
                'acc': f'{train_correct / train_total:.3f}'
            })

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for X, y in dl_val:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        val_acc = (val_preds == val_labels).mean()

        # 클래스별 정확도
        short_mask = val_labels == 0
        long_mask = val_labels == 1
        short_acc = (val_preds[short_mask] == val_labels[short_mask]).mean() if short_mask.sum() > 0 else 0
        long_acc = (val_preds[long_mask] == val_labels[long_mask]).mean() if long_mask.sum() > 0 else 0
        bal_acc = (short_acc + long_acc) / 2

        # 예측 분포
        n_pred_short = (val_preds == 0).sum()
        n_pred_long = (val_preds == 1).sum()

        print(f"[{epoch + 1:02d}] "
              f"loss={train_loss / train_total:.4f} "
              f"val_loss={val_loss / len(val_preds):.4f} "
              f"val_acc={val_acc:.3f} "
              f"bal_acc={bal_acc:.3f} "
              f"[S:{short_acc:.2f} L:{long_acc:.2f}] "
              f"[pred S:{n_pred_short} L:{n_pred_long}]",
              end='')

        scheduler.step()

        # 저장
        if bal_acc > best_bal_acc:
            best_acc = val_acc
            best_bal_acc = bal_acc
            patience_counter = 0

            # 체크포인트 저장
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = {
                'model': model.state_dict(),
                'feat_cols': selected_feats,
                'scaler_mu': ds_train.mu,
                'scaler_sd': ds_train.sd,
                'meta': {
                    'seq_len': args.seq_len,
                    'patch_size': args.patch_size,
                    'd_model': args.d_model,
                    'n_heads': args.n_heads,
                    'n_layers': args.n_layers,
                    'horizon': args.horizon,
                    'min_bps': args.min_bps
                }
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'patchtst_best.ckpt'))
            print(f"  ✅ Best (bal_acc={bal_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n[EARLY STOP] No improvement for {args.patience} epochs")
                break

    print(f"\n[DONE] Best Val Acc: {best_acc:.3f}, Balanced Acc: {best_bal_acc:.3f}")
    print(f"[SAVED] {args.out_dir}/patchtst_best.ckpt")

    # 메타 저장
    meta = {
        'seq_len': args.seq_len,
        'horizon': args.horizon,
        'min_bps': args.min_bps,
        'feat_cols': selected_feats,
        'scaler_mu': ds_train.mu.tolist(),
        'scaler_sd': ds_train.sd.tolist()
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ==================== Main ====================
def parse_args():
    ap = argparse.ArgumentParser(description='PatchTST Training')

    # 데이터
    ap.add_argument('--data', required=True, help='Data file (.parquet or .csv)')
    ap.add_argument('--sample_ratio', type=float, default=0.3, help='Sample ratio (0.3 = 30%)')

    # 레이블
    ap.add_argument('--horizon', type=int, default=72, help='Prediction horizon (candles)')
    ap.add_argument('--min_bps', type=int, default=150, help='Min profit (bps)')
    ap.add_argument('--slippage', type=int, default=10, help='Slippage (bps)')

    # Feature
    ap.add_argument('--feature_selection', action='store_true', help='Enable feature selection')
    ap.add_argument('--top_k', type=int, default=20, help='Top K features')

    # 모델
    ap.add_argument('--seq_len', type=int, default=72, help='Sequence length')
    ap.add_argument('--patch_size', type=int, default=12, help='Patch size')
    ap.add_argument('--d_model', type=int, default=128, help='Model dimension')
    ap.add_argument('--n_heads', type=int, default=4, help='Attention heads')
    ap.add_argument('--n_layers', type=int, default=3, help='Transformer layers')
    ap.add_argument('--dropout', type=float, default=0.2, help='Dropout')

    # 학습
    ap.add_argument('--epochs', type=int, default=50, help='Epochs')
    ap.add_argument('--batch', type=int, default=256, help='Batch size')
    ap.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    ap.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # 출력
    ap.add_argument('--out_dir', type=str, default='./models_patchtst', help='Output directory')

    return ap.parse_args()


def main():
    args = parse_args()

    print("PatchTST Training Script")
    print("=" * 80)

    # 데이터 로드
    print(f"\n[LOAD] Loading {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    # 날짜 변환
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if not is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    else:
        if is_datetime64tz_dtype(df['date']):
            df['date'] = df['date'].dt.tz_convert('UTC')
        else:
            df['date'] = df['date'].dt.tz_localize('UTC')

    df = df.dropna(subset=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    print(f"  ✅ Loaded {len(df):,} candles")

    # 학습
    train_model(df, args)


if __name__ == '__main__':
    main()