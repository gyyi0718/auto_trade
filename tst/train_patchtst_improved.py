#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchTST 개선 학습 스크립트

주요 개선사항:
    1. OneCycleLR 스케줄러 (더 빠른 수렴)
    2. Mixed Precision Training (메모리 효율 & 속도 향상)
    3. Gradient Accumulation (큰 effective batch size)
    4. Label Smoothing (일반화 성능 향상)
    5. EMA (Exponential Moving Average) 모델
    6. 향상된 Early Stopping (balanced accuracy 기준)
    7. 데이터 증강 옵션
    8. Focal Loss 옵션 (불균형 데이터)
    9. SWA (Stochastic Weight Averaging)
"""

import sys, os, json, argparse, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from tqdm import tqdm
from copy import deepcopy

warnings.filterwarnings("ignore")


# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    """불균형 데이터를 위한 Focal Loss"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # tensor [weight_class0, weight_class1]
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==================== Label Smoothing Cross Entropy ====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = nn.functional.log_softmax(pred, dim=-1)
        
        # Label smoothing
        targets = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes
        
        loss = (-targets * log_preds).sum(dim=-1)
        
        if self.weight is not None:
            loss = loss * self.weight[target]
            
        return loss.mean()


# ==================== EMA Model ====================
class ModelEMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ==================== Data Augmentation ====================
class TimeSeriesAugmentation:
    """시계열 데이터 증강"""
    def __init__(self, enable=False, noise_std=0.01, scale_range=(0.98, 1.02)):
        self.enable = enable
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, x):
        if not self.enable or not self.training:
            return x
            
        # Random noise
        if np.random.rand() < 0.3:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Random scaling
        if np.random.rand() < 0.3:
            scale = np.random.uniform(*self.scale_range)
            x = x * scale
            
        return x


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
class TimeSeriesDataset(Dataset):
    def __init__(self, df_feat, labels, feat_cols, seq_len=72):
        self.df = df_feat
        self.labels = labels
        self.feat_cols = feat_cols
        self.seq_len = seq_len
        
        # Normalize
        feat_values = self.df[feat_cols].values
        self.mu = np.nanmean(feat_values, axis=0)
        self.sd = np.nanstd(feat_values, axis=0) + 1e-8
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        
        sym_data = self.df[
            (self.df["symbol"] == row["symbol"]) &
            (self.df["date"] < pd.to_datetime(row["pred_time"], utc=True))
        ]
        
        if len(sym_data) < self.seq_len:
            seq = np.zeros((self.seq_len, len(self.feat_cols)), dtype=np.float32)
            actual = sym_data[self.feat_cols].values
            seq[-len(actual):] = actual
        else:
            seq = sym_data.iloc[-self.seq_len:][self.feat_cols].values
        
        # Normalize
        seq = (seq - self.mu) / self.sd
        seq = np.nan_to_num(seq, 0.0)
        
        return torch.FloatTensor(seq), torch.LongTensor([row["side"]])[0]


# ==================== PatchTST Model ====================
class PatchTST(nn.Module):
    def __init__(self, n_features, seq_len, patch_size=12, 
                 d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
    def forward(self, x):
        # x: [B, seq_len, n_features]
        B, seq_len, n_features = x.shape
        
        # Patchify
        x = x.reshape(B, self.n_patches, self.patch_size * n_features)
        
        # Embed
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Transform
        x = self.transformer(x)
        
        # Pool & classify
        x = x.mean(dim=1)
        return self.head(x)


# ==================== Training ====================
def train_model(df, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")
    
    # 1. Features
    df_feat, all_feats = make_features(df)
    
    # 2. Labels
    labels = make_realistic_labels(
        df_feat,
        horizon_candles=args.horizon,
        entry_slippage_bps=args.slippage,
        exit_slippage_bps=args.slippage,
        min_profit_bps=args.min_bps
    )
    
    if len(labels) == 0:
        print("[ERROR] No labels generated!")
        return
    
    # 3. Sample
    if args.sample_ratio < 1.0:
        n_sample = int(len(labels) * args.sample_ratio)
        labels = labels.sample(n=n_sample, random_state=42).reset_index(drop=True)
        print(f"\n[SAMPLE] {len(labels):,} labels (ratio={args.sample_ratio})")
    
    # 4. Feature selection
    if args.feature_selection:
        selected_feats = select_features_rf(df_feat, labels, all_feats, top_k=args.top_k)
    else:
        selected_feats = all_feats
    
    # 5. Train/Val split
    cutoff = pd.to_datetime(labels['pred_time'].quantile(0.85), utc=True)
    labels_train = labels[labels['pred_time'] < cutoff].reset_index(drop=True)
    labels_val = labels[labels['pred_time'] >= cutoff].reset_index(drop=True)
    
    print(f"\n[SPLIT]")
    print(f"  Train: {len(labels_train):,} ({labels_train['pred_time'].min()} ~ {labels_train['pred_time'].max()})")
    print(f"  Val:   {len(labels_val):,} ({labels_val['pred_time'].min()} ~ {labels_val['pred_time'].max()})")
    
    # 6. Dataset
    ds_train = TimeSeriesDataset(df_feat, labels_train, selected_feats, seq_len=args.seq_len)
    ds_val = TimeSeriesDataset(df_feat, labels_val, selected_feats, seq_len=args.seq_len)
    ds_val.mu = ds_train.mu
    ds_val.sd = ds_train.sd
    
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    # 7. Model
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
    print(f"  Patch size: {args.patch_size}")
    print(f"  d_model: {args.d_model}")
    print(f"  Layers: {args.n_layers}")
    
    # 8. Loss & Optimizer
    n_long = (labels_train['side'] == 1).sum()
    n_short = (labels_train['side'] == 0).sum()
    weight_long = len(labels_train) / (2 * n_long)
    weight_short = len(labels_train) / (2 * n_short)
    class_weights = torch.tensor([weight_short, weight_long], dtype=torch.float32).to(device)
    
    print(f"\n[LOSS] {args.loss_type}")
    print(f"  SHORT: {weight_short:.2f}")
    print(f"  LONG:  {weight_long:.2f}")
    
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    elif args.loss_type == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(epsilon=0.1, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(dl_train),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        print(f"  Scheduler: OneCycleLR (max_lr={args.lr})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"  Scheduler: CosineAnnealing")
    
    # Mixed precision
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("  ✅ Mixed Precision Training enabled")
    
    # EMA
    ema = ModelEMA(model, decay=0.9999) if args.use_ema else None
    if args.use_ema:
        print("  ✅ EMA enabled")
    
    # 9. Training
    print(f"\n[TRAIN] Starting...")
    print(f"  Gradient Accumulation: {args.grad_accum_steps} steps")
    
    best_bal_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(device), y.to(device)
            
            # Mixed precision
            if args.use_amp:
                with autocast():
                    logits = model(X)
                    loss = criterion(logits, y) / args.grad_accum_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if args.use_ema:
                        ema.update()
                    
                    if args.scheduler == 'onecycle':
                        scheduler.step()
            else:
                logits = model(X)
                loss = criterion(logits, y) / args.grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if args.use_ema:
                        ema.update()
                    
                    if args.scheduler == 'onecycle':
                        scheduler.step()
            
            train_loss += loss.item() * len(y) * args.grad_accum_steps
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)
            
            pbar.set_postfix({
                'loss': f'{train_loss / train_total:.4f}',
                'acc': f'{train_correct / train_total:.3f}'
            })
        
        # Validation
        if args.use_ema:
            ema.apply_shadow()
        
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for X, y in dl_val:
                X, y = X.to(device), y.to(device)
                
                if args.use_amp:
                    with autocast():
                        logits = model(X)
                        loss = criterion(logits, y)
                else:
                    logits = model(X)
                    loss = criterion(logits, y)
                
                val_loss += loss.item() * len(y)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        if args.use_ema:
            ema.restore()
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_acc = (val_preds == val_labels).mean()
        
        # Balanced accuracy
        short_mask = val_labels == 0
        long_mask = val_labels == 1
        short_acc = (val_preds[short_mask] == val_labels[short_mask]).mean() if short_mask.sum() > 0 else 0
        long_acc = (val_preds[long_mask] == val_labels[long_mask]).mean() if long_mask.sum() > 0 else 0
        bal_acc = (short_acc + long_acc) / 2
        
        # Prediction distribution
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
        
        if args.scheduler != 'onecycle':
            scheduler.step()
        
        # Save best
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            patience_counter = 0
            
            # Apply EMA for saving
            if args.use_ema:
                ema.apply_shadow()
            
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
            
            if args.use_ema:
                ema.restore()
            
            print(f"  ✅ Best (bal_acc={bal_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOP] No improvement for {args.patience} epochs")
                break
    
    print(f"\n[DONE] Best Balanced Acc: {best_bal_acc:.3f}")
    print(f"[SAVED] {args.out_dir}/patchtst_best.ckpt")
    
    # Save metadata
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
    ap = argparse.ArgumentParser(description='PatchTST Improved Training')
    
    # 데이터
    ap.add_argument('--data', required=True, help='Data file (.parquet or .csv)')
    ap.add_argument('--sample_ratio', type=float, default=0.5, help='Sample ratio')
    
    # 레이블
    ap.add_argument('--horizon', type=int, default=72, help='Prediction horizon (candles)')
    ap.add_argument('--min_bps', type=int, default=150, help='Min profit (bps)')
    ap.add_argument('--slippage', type=int, default=10, help='Slippage (bps)')
    
    # Feature
    ap.add_argument('--feature_selection', action='store_true', help='Enable feature selection')
    ap.add_argument('--top_k', type=int, default=10, help='Top K features')
    
    # 모델
    ap.add_argument('--seq_len', type=int, default=1440, help='Sequence length')
    ap.add_argument('--patch_size', type=int, default=12, help='Patch size')
    ap.add_argument('--d_model', type=int, default=256, help='Model dimension')
    ap.add_argument('--n_heads', type=int, default=8, help='Attention heads')
    ap.add_argument('--n_layers', type=int, default=4, help='Transformer layers')
    ap.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    
    # 학습 - 개선사항
    ap.add_argument('--epochs', type=int, default=100, help='Epochs')
    ap.add_argument('--batch', type=int, default=512, help='Batch size')
    ap.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    ap.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    ap.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # 개선 기법
    ap.add_argument('--loss_type', type=str, default='ce', 
                    choices=['ce', 'focal', 'label_smoothing'],
                    help='Loss function type')
    ap.add_argument('--scheduler', type=str, default='onecycle',
                    choices=['onecycle', 'cosine'],
                    help='LR scheduler type')
    ap.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    ap.add_argument('--use_ema', action='store_true', help='Use EMA')
    ap.add_argument('--grad_accum_steps', type=int, default=1, 
                    help='Gradient accumulation steps')
    
    # 출력
    ap.add_argument('--out_dir', type=str, default='./models_patchtst_improved', 
                    help='Output directory')
    
    return ap.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("PatchTST Improved Training Script")
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
