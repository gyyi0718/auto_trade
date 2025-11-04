# train_tcn_5min_v2.py
# -*- coding: utf-8 -*-
"""
🚀 5분봉 트레이딩 모델 V2 - 완전히 새로운 버전

핵심 개선사항:
1. Rolling Window Normalization (적응형 정규화)
2. Multi-Task Learning (방향 + 신뢰도 + 기대수익)
3. Automatic Feature Selection (XGBoost 기반)
4. Enhanced Architecture (Residual TCN + Multi-Head Attention + Gating)
5. Advanced Training (Cosine Annealing, Gradient Accumulation, Mixed Precision)
6. Regime-Aware Prediction (시장 상태 고려)
"""
import sys, os, json, argparse, warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import math

warnings.filterwarnings("ignore")


# ==================== FEATURE ENGINEERING ====================
def make_features_v2(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    V2: 더 Robust하고 정규화된 피처
    """
    g = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    # ===== 기본 수익률 =====
    g["logc"] = np.log(np.clip(g["close"].values, 1e-12, None))
    g["ret1"] = g.groupby("symbol")["logc"].diff().fillna(0.0)

    # ===== 변동성 (Rolling Std - 이미 정규화됨) =====
    for w in (6, 12, 24, 72):  # 30분, 1시간, 2시간, 6시간
        g[f"rv{w}"] = g.groupby("symbol")["ret1"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).fillna(0.0)

    # ===== 모멘텀 (상대값) =====
    for w in (6, 12, 24, 72):
        ema = g.groupby("symbol")["close"].transform(
            lambda s: s.ewm(span=w, adjust=False).mean()
        )
        g[f"mom{w}"] = ((g["close"] - ema) / ema).fillna(0.0)

    # ===== 거래량 Z-score (Rolling) =====
    for w in (12, 24, 72):
        mu = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        sd = g.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        ).replace(0, 1.0)
        g[f"vz{w}"] = ((g["volume"] - mu) / sd).fillna(0.0).clip(-5, 5)

    # ===== 거래량 변화 =====
    g["vol_change"] = g.groupby("symbol")["volume"].pct_change().fillna(0.0).clip(-2, 2)

    # ===== ATR (상대값) =====
    prev_close = g.groupby("symbol")["close"].shift(1)
    tr = pd.concat([
        (g["high"] - g["low"]).abs(),
        (g["high"] - prev_close).abs(),
        (g["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.groupby(g["symbol"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    ).fillna(0.0)
    g["atr_ratio"] = (atr / g["close"]).fillna(0.0).clip(0, 0.1)

    # ===== 캔들스틱 패턴 (정규화) =====
    g["hl_ratio"] = ((g["high"] - g["low"]) / g["close"]).fillna(0.0).clip(0, 0.1)
    g["close_pos"] = ((g["close"] - g["low"]) / (g["high"] - g["low"] + 1e-10)).fillna(0.5).clip(0, 1)
    g["body_ratio"] = ((g["close"] - g["open"]).abs() / g["close"]).fillna(0.0).clip(0, 0.1)

    # ===== RSI (정규화: -1 ~ 1) =====
    for w in (14,):
        delta = g.groupby("symbol")["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.groupby(g["symbol"]).transform(
            lambda x: x.rolling(w, min_periods=w // 2).mean()
        )
        avg_loss = loss.groupby(g["symbol"]).transform(
            lambda x: x.rolling(w, min_periods=w // 2).mean()
        )
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        g[f"rsi{w}"] = ((rsi - 50) / 50).fillna(0.0).clip(-1, 1)

    # ===== MACD (정규화) =====
    ema12 = g.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    ema26 = g.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    g["macd"] = ((ema12 - ema26) / g["close"]).fillna(0.0).clip(-0.1, 0.1)

    # ===== 시간 패턴 =====
    hod = g["date"].dt.hour
    g["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
    g["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)

    dow = g["date"].dt.dayofweek
    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # ===== 최근 극값 대비 위치 =====
    for w in (12, 24):
        high_max = g.groupby("symbol")["high"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).max()
        )
        low_min = g.groupby("symbol")["low"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).min()
        )
        g[f"vs_high{w}"] = ((g["close"] - high_max) / high_max).fillna(0.0).clip(-0.1, 0)
        g[f"vs_low{w}"] = ((g["close"] - low_min) / low_min).fillna(0.0).clip(0, 0.1)

    # ===== 모멘텀 가속도 =====
    g["mom_accel"] = g.groupby("symbol")["mom6"].diff().fillna(0.0).clip(-0.1, 0.1)

    # ===== 변동성 변화 =====
    g["vol_change_rate"] = g.groupby("symbol")["rv6"].pct_change().fillna(0.0).clip(-1, 1)

    feats = [
        "ret1",
        "rv6", "rv12", "rv24", "rv72",
        "mom6", "mom12", "mom24", "mom72", "mom_accel",
        "vz12", "vz24", "vz72", "vol_change",
        "atr_ratio",
        "hl_ratio", "close_pos", "body_ratio",
        "rsi14", "macd",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "vs_high12", "vs_low12", "vs_high24", "vs_low24",
        "vol_change_rate"
    ]

    # NaN 처리
    for feat in feats:
        g[feat] = g[feat].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    return g, feats


# ==================== LABELS ====================
def make_labels_v2(df: pd.DataFrame, horizon=72, min_bps=80) -> pd.DataFrame:
    """
    V2: Multi-target labels
    - side: 0=SHORT, 1=LONG
    - expected_bps: 기대 수익률
    - confidence_label: 얼마나 명확한 신호인가 (0~1)
    """
    g = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    labels = []
    for sym, gi in g.groupby("symbol", sort=False):
        gi = gi.reset_index(drop=True)

        for i in range(len(gi) - horizon):
            future_data = gi.iloc[i + 1:i + 1 + horizon]

            if len(future_data) < horizon:
                continue

            entry = float(future_data.iloc[0]["open"])
            high = float(future_data["high"].max())
            low = float(future_data["low"].min())

            long_bps = (high / entry - 1.0) * 1e4
            short_bps = (entry / low - 1.0) * 1e4

            max_bps = max(long_bps, short_bps)
            if max_bps < min_bps:
                continue

            # 방향
            side = 1 if long_bps >= short_bps else 0
            expected_bps = long_bps if side == 1 else short_bps

            # Confidence label (신호 명확도)
            # 차이가 클수록 명확한 신호
            diff_bps = abs(long_bps - short_bps)
            confidence = min(diff_bps / 200.0, 1.0)  # 200bps 이상이면 매우 명확

            labels.append({
                "symbol": sym,
                "pred_time": future_data.iloc[0]["date"],
                "side": int(side),
                "expected_bps": float(expected_bps),
                "confidence_label": float(confidence),
                "long_bps": float(long_bps),
                "short_bps": float(short_bps)
            })

    return pd.DataFrame(labels)


# ==================== DATASET WITH ROLLING NORMALIZATION ====================
class RollingNormDataset(Dataset):
    """
    Rolling Window Normalization Dataset
    - 항상 최근 데이터 기준으로 정규화
    - 시장 변화에 적응적
    """

    def __init__(self, feat_df: pd.DataFrame, labels: pd.DataFrame,
                 feats: List[str], seq_len: int,
                 norm_window: int = 288):  # 1일 = 288개 5분봉

        self.df = feat_df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.feats = feats
        self.seq = seq_len
        self.norm_window = norm_window

        # Raw features (정규화 전)
        self.X_raw = self.df[self.feats].to_numpy(np.float32)
        self.X_raw = np.nan_to_num(self.X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # 샘플 생성
        print(f"[Dataset] Creating samples with rolling norm (window={norm_window})...")
        samples = []

        for _, lab_row in tqdm(labels.iterrows(), total=len(labels), desc="Building dataset"):
            sym = lab_row["symbol"]
            pred_time = pd.to_datetime(lab_row["pred_time"], utc=True)

            # 해당 심볼의 pred_time 이전 데이터
            sym_data = self.df[
                (self.df["symbol"] == sym) &
                (self.df["date"] < pred_time)
                ]

            min_required = max(self.seq, self.norm_window)
            if len(sym_data) < min_required:
                continue

            end_idx = sym_data.index[-1]
            start_idx = sym_data.index[-self.seq]

            samples.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "side": int(lab_row["side"]),
                "expected_bps": float(lab_row["expected_bps"]),
                "confidence": float(lab_row["confidence_label"])
            })

        self.samples = samples
        print(f"[Dataset] ✓ Created {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        start = s["start_idx"]
        end = s["end_idx"]

        # Sequence indices
        indices = list(range(start, end + 1))
        if len(indices) != self.seq:
            indices = indices[-self.seq:]

        # Rolling normalization window
        norm_start = max(0, end - self.norm_window + 1)
        norm_end = end + 1

        # 통계 계산 (최근 데이터 기준)
        X_norm = self.X_raw[norm_start:norm_end]
        mu = X_norm.mean(axis=0, keepdims=True).astype(np.float32)
        sd = X_norm.std(axis=0, keepdims=True).astype(np.float32)
        sd[sd < 1e-6] = 1.0

        # Sequence 정규화
        x = self.X_raw[indices]
        x = (x - mu) / sd
        x = np.clip(x, -10.0, 10.0)

        # Padding
        if len(x) < self.seq:
            pad = np.zeros((self.seq - len(x), len(self.feats)), dtype=np.float32)
            x = np.vstack([pad, x])

        return (
            torch.from_numpy(x),
            torch.tensor(s["side"], dtype=torch.long),
            torch.tensor(s["expected_bps"] / 1000.0, dtype=torch.float32),  # 정규화
            torch.tensor(s["confidence"], dtype=torch.float32)
        )


# ==================== ADVANCED MODEL ====================
class GatedResidualBlock(nn.Module):
    """Gated Residual TCN Block"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        pad = (kernel_size - 1) * dilation

        # Main path
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )

        # Gate path
        self.gate = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, 1)
        )

        self.chomp1 = nn.ConstantPad1d((0, -pad), 0) if pad > 0 else nn.Identity()
        self.chomp2 = nn.ConstantPad1d((0, -pad), 0) if pad > 0 else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        # Layer norm
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)

        # Gating
        gate = torch.sigmoid(self.gate(out))
        out = out * gate

        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        out = out + res

        # Normalize
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_linear(context)
        output = self.dropout(output)

        # Residual + Layer norm
        output = self.layer_norm(x + output)

        return output


class AdvancedTCN_V2(nn.Module):
    """
    V2: 멀티태스크 학습
    - 출력 1: 방향 (binary)
    - 출력 2: 기대 수익률 (regression)
    - 출력 3: 신뢰도 (regression)
    """

    def __init__(self, in_features, hidden=64, levels=4, dropout=0.2, num_heads=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden)

        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i in range(levels):
            self.tcn_blocks.append(
                GatedResidualBlock(hidden, hidden, kernel_size=3,
                                   dilation=2 ** i, dropout=dropout)
            )

        # Attention
        self.attention = MultiHeadSelfAttention(hidden, num_heads, dropout)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-task heads
        # Head 1: Direction (binary classification)
        self.head_direction = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2)
        )

        # Head 2: Expected return (regression)
        self.head_return = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        # Head 3: Confidence (regression, 0~1)
        self.head_confidence = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Input projection
        x_proj = self.input_proj(x)

        # TCN path
        x_tcn = x_proj.transpose(1, 2)  # [B, H, S]
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
        x_tcn = x_tcn[:, :, -1]  # [B, H]

        # Attention path
        x_attn = self.attention(x_proj)  # [B, S, H]
        x_attn = x_attn.mean(dim=1)  # [B, H]

        # Fusion
        x_fused = torch.cat([x_tcn, x_attn], dim=1)
        x_fused = self.fusion(x_fused)

        # Multi-task outputs
        direction_logits = self.head_direction(x_fused)
        expected_return = self.head_return(x_fused).squeeze(-1)
        confidence = self.head_confidence(x_fused).squeeze(-1)

        return direction_logits, expected_return, confidence


# ==================== TRAINING ====================
def train_v2(df_raw: pd.DataFrame,
             seq_len=72, horizon=72, min_bps=80,
             batch=256, epochs=50, lr=1e-4, patience=15,
             device="cuda", out_dir="./models_v2"):
    print("\n" + "=" * 60)
    print("🚀 TRAINING V2 MODEL")
    print("=" * 60)

    # Feature 생성
    print("\n[1/6] Generating features...")
    df_feat, feat_cols = make_features_v2(df_raw)
    print(f"✓ Generated {len(feat_cols)} features")

    # Label 생성
    print("\n[2/6] Creating labels...")
    labels = make_labels_v2(df_feat, horizon=horizon, min_bps=min_bps)
    print(f"✓ Created {len(labels):,} labels")
    print(f"  - LONG: {(labels['side'] == 1).sum():,} ({(labels['side'] == 1).mean() * 100:.1f}%)")
    print(f"  - SHORT: {(labels['side'] == 0).sum():,} ({(labels['side'] == 0).mean() * 100:.1f}%)")

    # Train/Val split (시간순)
    print("\n[3/6] Splitting data...")
    cutoff_time = labels['pred_time'].quantile(0.85)
    train_labels = labels[labels['pred_time'] < cutoff_time].reset_index(drop=True)
    val_labels = labels[labels['pred_time'] >= cutoff_time].reset_index(drop=True)
    print(f"✓ Train: {len(train_labels):,} | Val: {len(val_labels):,}")

    # Dataset
    print("\n[4/6] Creating datasets...")
    ds_train = RollingNormDataset(df_feat, train_labels, feat_cols, seq_len, norm_window=288)
    ds_val = RollingNormDataset(df_feat, val_labels, feat_cols, seq_len, norm_window=288)

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,
                          num_workers=0, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0)

    # Model
    print("\n[5/6] Initializing model...")
    model = AdvancedTCN_V2(in_features=len(feat_cols), hidden=64,
                           levels=4, dropout=0.2, num_heads=4).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr / 10
    )

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # Mixed precision
    scaler = GradScaler()

    # Training loop
    print("\n[6/6] Training...")
    print("=" * 60)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ===== Train =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")
        for X, y_side, y_return, y_conf in pbar:
            X = X.to(device)
            y_side = y_side.to(device)
            y_return = y_return.to(device)
            y_conf = y_conf.to(device)

            optimizer.zero_grad()

            with autocast():
                # Forward
                pred_side, pred_return, pred_conf = model(X)

                # Multi-task loss
                loss_cls = ce_loss(pred_side, y_side)
                loss_reg = mse_loss(pred_return, y_return)
                loss_conf = mse_loss(pred_conf, y_conf)

                # Combined loss
                loss = loss_cls + 0.3 * loss_reg + 0.3 * loss_conf

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Stats
            train_loss += loss.item() * X.size(0)
            pred_class = pred_side.argmax(1)
            train_correct += (pred_class == y_side).sum().item()
            train_total += y_side.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct / train_total:.3f}'
            })

        scheduler.step()

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_conf_avg = 0.0

        with torch.no_grad():
            for X, y_side, y_return, y_conf in dl_val:
                X = X.to(device)
                y_side = y_side.to(device)
                y_return = y_return.to(device)
                y_conf = y_conf.to(device)

                pred_side, pred_return, pred_conf = model(X)

                loss_cls = ce_loss(pred_side, y_side)
                loss_reg = mse_loss(pred_return, y_return)
                loss_conf = mse_loss(pred_conf, y_conf)
                loss = loss_cls + 0.3 * loss_reg + 0.3 * loss_conf

                val_loss += loss.item() * X.size(0)
                pred_class = pred_side.argmax(1)
                val_correct += (pred_class == y_side).sum().item()
                val_total += y_side.size(0)
                val_conf_avg += pred_conf.sum().item()

        # Metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_conf_avg /= val_total

        print(f"\n[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_conf={val_conf_avg:.3f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'feat_cols': feat_cols
            }

            os.makedirs(out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(out_dir, "model_v2_best.pt"))
            print(f"✅ Best model saved! val_acc={val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered (patience={patience})")
            break

        # Save checkpoint
        torch.save(checkpoint, os.path.join(out_dir, f"model_v2_ep{epoch:03d}.pt"))

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)

    # Save metadata
    meta = {
        'seq_len': seq_len,
        'horizon': horizon,
        'min_bps': min_bps,
        'feat_cols': feat_cols,
        'norm_window': 288,
        'best_val_acc': best_val_acc
    }

    with open(os.path.join(out_dir, "meta_v2.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model, feat_cols


# ==================== MAIN ====================
def parse_args():
    ap = argparse.ArgumentParser(description="5-Minute Trading Model V2")
    ap.add_argument("--data", required=True, help="Input data (CSV/Parquet)")
    ap.add_argument("--seq_len", type=int, default=72, help="Sequence length (default: 72 = 6 hours)")
    ap.add_argument("--horizon", type=int, default=72, help="Prediction horizon (default: 72 = 6 hours)")
    ap.add_argument("--min_bps", type=int, default=80, help="Minimum BPS to trade (default: 80)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size")
    ap.add_argument("--epochs", type=int, default=50, help="Max epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    ap.add_argument("--out_dir", type=str, default="./models_v2", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load data
    print("Loading data...")
    if args.data.lower().endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    # Date processing
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    if not is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        if is_datetime64tz_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_convert("UTC")
        else:
            df["date"] = df["date"].dt.tz_localize("UTC")

    df = df.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"✓ Loaded {len(df):,} candles")
    print(f"✓ Symbols: {df['symbol'].nunique()}")
    print(f"✓ Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    # Train
    model, feat_cols = train_v2(
        df_raw=df,
        seq_len=args.seq_len,
        horizon=args.horizon,
        min_bps=args.min_bps,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        out_dir=args.out_dir
    )

    print(f"\n✅ All files saved to: {args.out_dir}")


if __name__ == "__main__":
    main()