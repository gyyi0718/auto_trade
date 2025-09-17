# -*- coding: utf-8 -*-
"""
Bybit Futures(USDT-Perp, category='linear') 1m 수집 + DeepAR 멀티시리즈 학습 (다심볼)
- 타깃: log_return (ln(close_t / close_{t-1}))
- 손실: Normal NLL + λ * sign-BCE(μ의 부호가 실제 수익률 부호와 일치하도록 가중)
- 인코더/예측 길이: SEQ_LEN=60, PRED_LEN=60

산출물:
    data/combined_1m.csv
    data/{symbol}_deepar_input.csv
    models/multi_deepar_best.ckpt
    models/multi_deepar_model.ckpt
    logs/deepar/version_*/
"""

import os, time, json, shutil, random, math, requests
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datetime import datetime, timedelta
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss, DistributionLoss
from pytorch_forecasting.data import GroupNormalizer
# ====================== Sign-aware Loss ======================


# ====================== 사용자 설정 ======================
SYMBOLS = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]



CATEGORY   = "linear"   # Bybit USDT-Perp
INTERVAL   = "1"        # 1m

# UTC 기준 기간
today = datetime.utcnow()
start_date = today - timedelta(days=720)
START_UTC = start_date.strftime("%Y-%m-%d %H:%M:%S")
END_UTC   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# DeepAR 하이퍼
SEQ_LEN    = int(os.getenv("SEQ_LEN", "60"))
PRED_LEN   = int(os.getenv("PRED_LEN","60"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS","100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE","256"))
LR_DEEPAR  = float(os.getenv("LR_DEEPAR","1e-3"))
HIDDEN     = int(os.getenv("HIDDEN","64"))
RNN_LAYERS = int(os.getenv("RNN_LAYERS","2"))
DROPOUT    = float(os.getenv("DROPOUT","0.1"))
EARLY_PATIENCE = int(os.getenv("EARLY_PATIENCE","10"))

# Sign penalty 계수
LAMBDA_SIGN = float(os.getenv("LAMBDA_SIGN","0.3"))  # 0.2~0.4 권장
ALPHA_SIGN  = float(os.getenv("ALPHA_SIGN","7.0"))   # 5~10 권장

# 경로
DIR_DATA   = "data"
DIR_MODELS = "models"
DIR_LOGS   = "logs"
os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_LOGS, exist_ok=True)

COMBINED_CSV   = os.path.join(DIR_DATA, "combined_1m.csv")
BEST_CKPT_PATH = os.path.join(DIR_MODELS, "multi_deepar_best.ckpt")
LIVE_CKPT_PATH = os.path.join(DIR_MODELS, "multi_deepar_model.ckpt")

# ====================== Bybit v5 ======================
BASE_URL = "https://api.bybit.com/v5/market/kline"

def to_ms(dt_utc_str: str) -> int:
    return int(pd.Timestamp(dt_utc_str, tz="UTC").timestamp() * 1000)

def fetch_ohlcv_linear_1m(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "category": CATEGORY,
            "symbol": symbol,
            "interval": INTERVAL,
            "start": cursor,
            "end": end_ms,
            "limit": limit
        }
        r = requests.get(BASE_URL, params=params, timeout=15)
        try:
            data = r.json()
        except Exception:
            print(f"[WARN] {symbol} json parse fail at {cursor}, status={r.status_code}")
            break

        res = data.get("result", {})
        rows = res.get("list", [])
        if not rows:
            break

        for row in rows:
            ts = int(row[0])
            all_rows.append({
                "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                "series_id": symbol,
                "open": float(row[1]),
                "high": float(row[2]),
                "low":  float(row[3]),
                "close":float(row[4]),
                "volume":float(row[5]),
                "turnover":float(row[6]),
            })

        max_ts = max(int(r[0]) for r in rows)
        if max_ts <= cursor:
            break
        cursor = max_ts + 60_000  # 다음 분부터
        time.sleep(0.08)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def build_combined_csv():
    start_ms = to_ms(START_UTC)
    end_ms   = to_ms(END_UTC)

    frames = []
    for sym in SYMBOLS:
        print(f"[INFO] Fetch {sym} ({START_UTC} ~ {END_UTC}, 1m, linear)")
        df = fetch_ohlcv_linear_1m(sym, start_ms, end_ms)
        if df.empty:
            print(f"[WARN] {sym} no data (linear futures).")
            continue

        # log_return 생성 (분단위)
        df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
        frames.append(df)

        # 심볼별 최신본
        latest_csv = os.path.join(DIR_DATA, f"{sym}_deepar_input.csv")
        df_out = df.copy()
        df_out["time_idx"] = np.arange(len(df_out), dtype=np.int64)
        cols = ["time_idx","series_id","timestamp","open","high","low","close","volume","log_return"]
        df_out[cols].to_csv(latest_csv, index=False, encoding="utf-8")
        print(f"  -> saved {latest_csv} ({len(df_out):,} rows)")

    if not frames:
        raise RuntimeError("No data collected for all symbols.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(COMBINED_CSV, index=False, encoding="utf-8")
    print(f"[DONE] combined saved: {COMBINED_CSV} ({len(all_df):,} rows)")
    return all_df



class SignAwareNormalLoss(NormalDistributionLoss):
    """
    base: Normal NLL
    extra: λ * BCEWithLogits(α·μ, 1[y>0])  # AMP 안전
    target 형태: y  또는  (y, weight)  또는 dict 포함
    """
    def __init__(self, reduction: str = "mean",
                 lambda_sign: float = 0.3,
                 alpha_sign: float = 7.0):
        super().__init__(reduction=reduction)
        self.lambda_sign = float(lambda_sign)
        self.alpha_sign  = float(alpha_sign)

    def _unpack_target(self, target):
        if isinstance(target, (tuple, list)):
            y = target[0]; w = target[1] if len(target) > 1 else None
        elif isinstance(target, dict):
            y = target.get("target", target.get("y", target))
            w = target.get("weight", None)
        else:
            y = target; w = None
        return y, w

    def forward(self, y_pred, target):
        # 1) 기본 NLL은 원형 유지
        base = super().forward(y_pred, target)

        # 2) sign 로짓 = α·μ  (to_prediction: 분포 평균 μ)
        y, w = self._unpack_target(target)
        mu   = self.to_prediction(y_pred)
        if mu.shape != y.shape:
            y = y.view_as(mu)

        logits = self.alpha_sign * mu
        y_pos  = (y > 0).float()

        # AMP 안전한 BCEWithLogits
        bce = F.binary_cross_entropy_with_logits(logits, y_pos, reduction="none")

        if w is not None:
            if w.shape != bce.shape:
                w = w.view_as(bce)
            denom = torch.clamp(w.sum(), min=1.0)
            bce = (bce * w).sum() / denom
        else:
            bce = bce.mean()

        return base + self.lambda_sign * bce

# ====================== DeepAR 멀티시리즈 학습 ======================
def train_deepar_multi(df_all: pd.DataFrame):
    df_all = df_all.sort_values(["series_id","timestamp"]).reset_index(drop=True)
    df_all["time_idx"] = df_all.groupby("series_id").cumcount()

    min_len = df_all.groupby("series_id")["time_idx"].max().min()
    if min_len < (SEQ_LEN + PRED_LEN + 10):
        print("[WARN] some series are short; consider shortening PRED_LEN or extending date range.")

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

    validation = TimeSeriesDataSet.from_dataset(
        training, df_all, predict=True, stop_randomization=True
    )

    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    print(f"[INFO] DeepAR train samples={len(train_loader.dataset):,}  "
          f"val samples={len(val_loader.dataset):,}  "
          f"groups={df_all['series_id'].nunique()}")

    loss_fn = SignAwareNormalLoss(lambda_sign=LAMBDA_SIGN, alpha_sign=ALPHA_SIGN)

    model = DeepAR.from_dataset(
        training,
        learning_rate=LR_DEEPAR,
        hidden_size=HIDDEN,
        rnn_layers=RNN_LAYERS,
        dropout=DROPOUT,
        loss=loss_fn
    )

    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="multi_deepar_best"
    )
    early = EarlyStopping(monitor="val_loss", patience=EARLY_PATIENCE, mode="min")

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early, ckpt_cb],
        logger=CSVLogger(DIR_LOGS, name="deepar"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        deterministic=True,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=50
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

# ====================== 실행부 ======================
if __name__ == "__main__":
    seed_everything(42, workers=True)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True

    df_all = build_combined_csv()
    train_deepar_multi(df_all)
    print("🎯 ALL DONE")
