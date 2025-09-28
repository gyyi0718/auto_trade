import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer
import pandas as pd
from datetime import datetime
# === 하이퍼파라미터 ===
SEQ_LEN = 60
PRED_LEN = 60
MAX_EPOCHS = 300
BATCH_SIZE = 256
symbol = "1000PEPEUSDT"
now = datetime.now()
date_str = now.strftime("%y%m%d")
# === 데이터 로딩 ===
df = pd.read_csv(f"{symbol}_deepar_input_{date_str}.csv")
df.dropna(inplace=True)

# === 학습/검증 분할 ===
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

# === DataLoader 생성 ===
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

# === 모델 수동 정의 (⚠️ 핵심) ===
model = DeepAR.from_dataset(
    training,  # <- TimeSeriesDataSet
    learning_rate=1e-3,
    hidden_size=32,
    rnn_layers=2,
    dropout=0.1,
    loss=NormalDistributionLoss()
)
# === 콜백 및 트레이너 ===
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

# === 학습 ===
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# === 저장 ===
trainer.save_checkpoint(f"{symbol}_deepar_model_{date_str}.ckpt")
print("✅ 모델 학습 및 저장 완료.")
