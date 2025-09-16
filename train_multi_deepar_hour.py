# train_deepar_hourly.py
# -*- coding: utf-8 -*-
import os, shutil, random
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer

# ====== 경로 / 하이퍼 ======
HOURLY_CSV = "bybit_futures_60.csv"  # ← 네가 가진 시간봉 합본(열: timestamp, series_id, open, high, low, close, volume ...)
DIR_MODELS = "models"; DIR_LOGS = "logs"
os.makedirs(DIR_MODELS, exist_ok=True); os.makedirs(DIR_LOGS, exist_ok=True)

SEQ_LEN    = 168   # 1주(168시간) 컨텍스트 권장
PRED_LEN   = 24    # ✅ 하루치(24개 1H 캔들) 예측
BATCH_SIZE = 256
MAX_EPOCHS = 100
LR         = 1e-3
HIDDEN     = 64
RNN_LAYERS = 2
DROPOUT    = 0.1
EARLY_PATIENCE = 10

BEST_CKPT_PATH = os.path.join(DIR_MODELS, "multi_deepar_best_hour.ckpt")
LIVE_CKPT_PATH = os.path.join(DIR_MODELS, "multi_deepar_model_hour.ckpt")

def load_hourly_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 심볼 컬럼명을 series_id로 변경
    if "symbol" in df.columns:
        df.rename(columns={"symbol": "series_id"}, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.sort_values(["series_id","timestamp"]).reset_index(drop=True)

    # 시간봉 기준 로그수익률
    df["log_return"] = (
        df.groupby("series_id")["close"]
          .apply(lambda s: np.log(s).diff())
          .reset_index(level=0, drop=True)
          .fillna(0.0)
    )

    # 시계열 인덱스
    df["time_idx"] = df.groupby("series_id").cumcount()
    return df


def train_deepar_multi(df_all: pd.DataFrame):
    # 최소 길이 체크 (선택)
    # if df_all.groupby("series_id")["time_idx"].max().min() < (SEQ_LEN + PRED_LEN + 10):
    #     print("[WARN] some series are short for given SEQ_LEN/PRED_LEN")

    global_max_t = df_all["time_idx"].max()
    train_cutoff = int(global_max_t - PRED_LEN)

    training = TimeSeriesDataSet(
        df_all[df_all.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df_all, predict=True, stop_randomization=True)
    train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    model = DeepAR.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=HIDDEN,
        rnn_layers=RNN_LAYERS,
        dropout=DROPOUT,
        loss=NormalDistributionLoss(),
    )

    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="multi_deepar_best")
    early   = EarlyStopping(monitor="val_loss", patience=EARLY_PATIENCE, mode="min")
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early, ckpt_cb],
        logger=CSVLogger(DIR_LOGS, name="deepar_hourly"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        deterministic=True,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copyfile(best_ckpt, BEST_CKPT_PATH)
        shutil.copyfile(best_ckpt, LIVE_CKPT_PATH)
        print(f"✅ DeepAR 저장 완료\n - best: {best_ckpt}\n - BEST: {BEST_CKPT_PATH}\n - LIVE: {LIVE_CKPT_PATH}")
    else:
        fallback = os.path.join(DIR_MODELS, "multi_deepar_fallback.ckpt")
        trainer.save_checkpoint(fallback)
        shutil.copyfile(fallback, LIVE_CKPT_PATH)
        print(f"✅ DeepAR 수동 저장 완료: {fallback} & {LIVE_CKPT_PATH}")

if __name__ == "__main__":
    seed_everything(42, workers=True)
    np.random.seed(42); random.seed(42); torch.backends.cudnn.benchmark = True
    df_hourly = load_hourly_df(HOURLY_CSV)
    print(df_hourly.groupby("series_id")["time_idx"].max().add(1).describe())  # 각 시리즈 길이 확인용
    train_deepar_multi(df_hourly)
