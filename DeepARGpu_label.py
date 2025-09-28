# -*- coding: utf-8 -*-
import os, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer

from datetime import datetime

# ===== 하이퍼파라미터 =====
SEQ_LEN     = 60
PRED_LEN    = 60
MAX_EPOCHS  = 300
BATCH_SIZE  = 256
LR_CLS      = 1e-3     # 분류기 학습률
EPOCHS_CLS  = 8        # 분류기 에폭

symbol   = "BTCUSDT"
now      = datetime.now()
date_str = now.strftime("%y%m%d")

# ===== 거래 비용/게이트 가정 (라벨 생성용) =====
TAKER_FEE_ONEWAY = 0.0004   # 0.04%  (편도)
SLIP_ONEWAY      = 0.0005   # 0.05%  (편도)
ROUNDTRIP        = 2 * (TAKER_FEE_ONEWAY + SLIP_ONEWAY)   # 0.0018 = 0.18%
LEVERAGE_LBL     = 10       # 라벨 산정에 사용할 레버리지 (게이트 기준)
ENTRY_PORTION    = 0.40     # 진입 비중
CAPITAL_ASSUME   = 1000.0   # 자본 가정 (최소이익 문턱 환산용)
MIN_PROFIT_USD   = 0.01     # 최소 기대 순이익

# ===== 데이터 로딩 =====
df = pd.read_csv(f"{symbol}_deepar_input_{date_str}.csv")
df.dropna(inplace=True)

# ------------------------------------------------------------------------------------------------
# (A) DeepAR: 원래 학습 (그대로 유지)
# ------------------------------------------------------------------------------------------------
train_cutoff = df["time_idx"].max() - PRED_LEN

training = TimeSeriesDataSet(
    df[df.time_idx <= train_cutoff],
    time_idx="time_idx",
    target="log_return",
    group_ids=["series_id"],
    max_encoder_length=SEQ_LEN,
    max_prediction_length=PRED_LEN,
    time_varying_known_reals=["time_idx", "volume"],
    time_varying_unknown_reals=["log_return"],
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

model = DeepAR.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=32,
    rnn_layers=2,
    dropout=0.1,
    loss=NormalDistributionLoss()
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename=f"{symbol}_best_deepar_model"
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5), checkpoint_callback],
    logger=CSVLogger("logs", name="deepar"),
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

trainer.save_checkpoint(f"{symbol}_deepar_model_{date_str}.ckpt")
print("✅ DeepAR 학습/저장 완료")

# ------------------------------------------------------------------------------------------------
# (B) Profit 게이트 분류기: 같은 CSV로 라벨 생성 + 간단 GRU 분류기 학습
#      - 라벨: 미래 1..PRED_LEN 중 |ROI|가 최대인 호라이즌 기준
#               -> 계정 기준 순ROI(레버리지/비용 반영)가 0 이상이고,
#               -> 최소 기대 순이익(MIN_PROFIT_USD)을 충족하면 1, 아니면 0
# ------------------------------------------------------------------------------------------------

def build_profit_dataset(df_src: pd.DataFrame,
                         seq_len=60, pred_len=60,
                         L=10,
                         fee_oneway=0.0004, slip_oneway=0.0005,
                         min_profit_usd=0.01, entry_portion=0.40, capital=1000.0):
    """
    같은 CSV에서 Profit 라벨을 생성해 (X, Y, meta)를 반환.
    - 각 시점 t에서 미래 1..pred_len 중 |ROI|가 최대인 호라이즌을 선택
    - 계정 기준 순ROI가 0 초과 & 최소이익 조건 충족이면 1, 아니면 0
    """
    df_ = df_src.copy().reset_index(drop=True)
    close = df_["close"].astype(float).values
    lr    = df_["log_return"].astype(float).values
    vol   = df_["volume"].astype(float).values if "volume" in df_.columns else np.zeros(len(df_), dtype=float)
    n     = len(df_)

    # 유효 라벨을 만들 수 있는 최대 행 수 (끝쪽 pred_len 구간은 미래가 부족)
    valid_n = n - pred_len
    if valid_n <= seq_len:
        raise RuntimeError("샘플이 너무 적습니다. seq_len/pred_len을 줄이거나 데이터 길이를 늘리세요.")

    # k-스텝 미래 수익률: ROI_k(t) = close[t+k]/close[t] - 1  (길이 항상 n-k)
    # valid 라벨 영역만 사용: 각 k마다 앞쪽 valid_n 길이만 취해 정렬
    roi_mat = np.full((valid_n, pred_len), np.nan, dtype=np.float64)  # (valid_n, pred_len)
    for k in range(1, pred_len + 1):
        roi_k = (close[k:] / close[:-k]) - 1.0           # (n-k,)
        roi_mat[:, k-1] = roi_k[:valid_n]                # (valid_n,)

    # 각 시점의 최적 호라이즌 선택
    abs_roi = np.abs(roi_mat)                            # (valid_n, pred_len)
    k_star  = abs_roi.argmax(axis=1)                     # (valid_n,)
    chosen  = roi_mat[np.arange(valid_n), k_star]        # (valid_n,) signed ROI

    # 라벨 계산(계정 기준)
    roundtrip = 2 * (fee_oneway + slip_oneway)
    fee_th_px = roundtrip / L
    extra_px  = (min_profit_usd / (capital * entry_portion * L)) if (capital*entry_portion*L) > 0 else 0.0
    px_gate   = fee_th_px + extra_px
    acct_net  = (L * np.abs(chosen)) - roundtrip
    y_full    = ((np.abs(chosen) >= px_gate) & (acct_net > 0)).astype(np.float32)  # (valid_n,)

    # 입력 피처: [log_return, volume_z, rolling_vol]
    lr_f = lr.astype(np.float32)
    vmu, vsd = (vol.mean() if vol.std() > 0 else 0.0), (vol.std() + 1e-8)
    vz  = ((vol - vmu) / vsd).astype(np.float32)
    rv  = pd.Series(lr_f).rolling(seq_len).std(ddof=0).fillna(0.0).values.astype(np.float32)

    feats = np.stack([lr_f, vz, rv], axis=1)  # (n, 3)

    # 시퀀스 샘플 생성: t ∈ [seq_len, valid_n-1]
    X_list, Y_list = [], []
    for t in range(seq_len, valid_n):
        X_list.append(feats[t-seq_len:t, :])  # (seq_len, 3)
        Y_list.append(y_full[t])              # 같은 시점의 라벨 사용

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, seq_len, 3)
    Y = np.array(Y_list, dtype=np.float32)            # (N,)
    return X, Y, {"px_gate": float(px_gate), "fee_th_px": float(fee_th_px), "extra_px": float(extra_px)}

class ProfitSeqDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)                           # (N, T, C)
        self.Y = torch.from_numpy(Y).unsqueeze(1).float()      # (N, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

class ProfitClassifier(nn.Module):
    def __init__(self, in_ch=3, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_ch, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)  # logit
        )
    def forward(self, x):
        out, _ = self.gru(x)         # (B, T, H)
        h = out[:, -1, :]            # (B, H)
        logit = self.head(h)         # (B, 1)
        return logit

def train_profit_classifier(X, Y, epochs=8, lr=1e-3, batch_size=256, save_path="profit_cls.pt"):
    # 80/20 split
    n = X.shape[0]
    idx = np.arange(n); rng = np.random.default_rng(42); rng.shuffle(idx)
    n_val = max(1, int(n * 0.2))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    ds_tr = ProfitSeqDS(X[tr_idx], Y[tr_idx])
    ds_va = ProfitSeqDS(X[val_idx], Y[val_idx])
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ProfitClassifier().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr)
    crit   = nn.BCEWithLogitsLoss()

    best = {"val": 1e9, "ep": -1}
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss  = crit(logit, yb)
            loss.backward(); opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= len(ds_tr)

        model.eval(); va_loss=0.0; corr=0; tot=0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb)
                loss  = crit(logit, yb)
                va_loss += float(loss.item()) * xb.size(0)
                pred = (torch.sigmoid(logit) >= 0.5).float()
                corr += float((pred==yb).sum().item())
                tot  += yb.numel()
        va_loss /= len(ds_va); acc = corr/max(1,tot)
        print(f"[Profit-CLS][EP {ep}] train_bce={tr_loss:.4f}  val_bce={va_loss:.4f}  val_acc={acc*100:.2f}%")

        if va_loss < best["val"]:
            best.update({"val": va_loss, "ep": ep})
            torch.save({"model": model.state_dict()}, save_path)
    print(f"[Profit-CLS] Best epoch={best['ep']}  val_bce={best['val']:.4f}")
    return save_path

# ==== 라벨 생성 + 학습 실행 ====
Xp, Yp, meta = build_profit_dataset(
    df, seq_len=SEQ_LEN, pred_len=PRED_LEN,
    L=LEVERAGE_LBL,
    fee_oneway=TAKER_FEE_ONEWAY, slip_oneway=SLIP_ONEWAY,
    min_profit_usd=MIN_PROFIT_USD, entry_portion=ENTRY_PORTION, capital=CAPITAL_ASSUME
)
cls_path = f"{symbol}_profit_cls.pt"
train_profit_classifier(Xp, Yp, epochs=EPOCHS_CLS, lr=LR_CLS, batch_size=BATCH_SIZE, save_path=cls_path)

with open(f"{symbol}_profit_cls_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "px_gate": meta["px_gate"],
        "fee_th_px": meta["fee_th_px"],
        "extra_px": meta["extra_px"],
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "leverage_for_label": LEVERAGE_LBL,
        "roundtrip": ROUNDTRIP,
        "entry_portion": ENTRY_PORTION,
        "capital_assume": CAPITAL_ASSUME,
        "min_profit_usd": MIN_PROFIT_USD
    }, f, ensure_ascii=False, indent=2)

print("✅ Profit 분류기 학습/저장 완료:", cls_path)
print("🎯 메타:", meta)
