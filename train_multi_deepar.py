# -*- coding: utf-8 -*-
"""
Bybit Futures(USDT-Perp, category='linear') 1m 수집 + DeepAR 멀티시리즈 학습 (11심볼)
- 입력: Bybit v5 /v5/market/kline
- 저장:
    data/combined_1m.csv                    (전체 합본)
    data/{symbol}_deepar_input.csv          (심볼별 최신본: 네 포맷에 맞춤)
- DeepAR 출력:
    models/multi_deepar_best.ckpt           (최고 성능)
    models/multi_deepar_model.ckpt          (실거래용 고정명)
    logs/deepar/version_*/                  (CSV 로그)
- 타깃: log_return (ln(close_t / close_{t-1}))
- 그룹: series_id = 심볼명 (예: "BTCUSDT")
- 인코더/예측 길이: SEQ_LEN=60, PRED_LEN=60  (원하면 바꿔)
"""

import os, time, math, json, shutil, random
import requests
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer
from datetime import datetime, timedelta

# ====================== 사용자 설정 ======================
SYMBOLS2 = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]

SYMBOLS = [
    "ETHUSDT"
]
CATEGORY   = "linear"   # Bybit USDT-Perp
INTERVAL   = "1"        # 1m
# UTC 기준 기간 지정

today = datetime.utcnow()
start_date = today - timedelta(days=720)
START_UTC = start_date.strftime("%Y-%m-%d %H:%M:%S")
END_UTC    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# DeepAR 하이퍼
SEQ_LEN    = 60
PRED_LEN   = 60
MAX_EPOCHS = 100
BATCH_SIZE = 256
LR_DEEPAR  = 1e-3
HIDDEN     = 64
RNN_LAYERS = 2
DROPOUT    = 0.1
EARLY_PATIENCE = 10

# 경로
DIR_DATA   = "data"
DIR_MODELS = "models"
DIR_LOGS   = "logs"

os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_LOGS, exist_ok=True)

# 고정 출력 파일명
COMBINED_CSV   = os.path.join(DIR_DATA, "combined_1m.csv")
BEST_CKPT_PATH = os.path.join(DIR_MODELS, "multi_deepar_best.ckpt")
LIVE_CKPT_PATH = os.path.join(DIR_MODELS, "multi_deepar_model.ckpt")

# ====================== Bybit v5 ======================
BASE_URL = "https://api.bybit.com/v5/market/kline"

def to_ms(dt_utc_str: str) -> int:
    return int(pd.Timestamp(dt_utc_str, tz="UTC").timestamp() * 1000)

def fetch_ohlcv_linear_1m(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """
    Bybit v5 /v5/market/kline (category='linear', interval='1')
    - 페이지네이션: start=cursor, end=end_ms, limit=1000
    - 응답 list는 보통 내림차순이지만, 안전하게 '가장 큰 ts'를 찾아 cursor 갱신
    """
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
            # 더 이상 없음
            break

        # rows: [startTime, open, high, low, close, volume, turnover]
        # 안심 정렬 및 수집
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
            # 무한루프 방지
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

        # 네가 쓰던 심볼별 최신본 파일도 저장
        latest_csv = os.path.join(DIR_DATA, f"{sym}_deepar_input.csv")
        df_out = df.copy()
        # time_idx: 심볼별 연속 인덱스
        df_out["time_idx"] = np.arange(len(df_out), dtype=np.int64)
        cols = ["time_idx","series_id","timestamp","open","high","low","close","volume","log_return"]
        df_out[cols].to_csv(latest_csv, index=False, encoding="utf-8")
        print(f"  -> saved {latest_csv} ({len(df_out):,} rows)")

    if not frames:
        raise RuntimeError("No data collected for all symbols (check symbols/date range).")

    all_df = pd.concat(frames, ignore_index=True)
    # 전 심볼 합본 CSV 저장
    all_df.to_csv(COMBINED_CSV, index=False, encoding="utf-8")
    print(f"[DONE] combined saved: {COMBINED_CSV} ({len(all_df):,} rows)")

    return all_df


# ====================== DeepAR 멀티시리즈 학습 ======================
def train_deepar_multi(df_all: pd.DataFrame):
    """
    - 타깃: log_return
    - 그룹: series_id
    - encoder: SEQ_LEN, prediction: PRED_LEN
    """
    # 심볼별 time_idx 부여
    df_all = df_all.sort_values(["series_id","timestamp"]).reset_index(drop=True)
    df_all["time_idx"] = df_all.groupby("series_id").cumcount()

    # 최소 길이 체크
    min_len = df_all.groupby("series_id")["time_idx"].max().min()
    if min_len < (SEQ_LEN + PRED_LEN + 10):
        print("[WARN] some series are too short; consider shortening PRED_LEN or date range.")

    # 학습/검증 커트오프: 가장 마지막 time_idx 기준
    # 각 시리즈마다 최대 time_idx가 다를 수 있으므로, 글로벌 max에서 PRED_LEN만큼 뺀 지점 사용
    global_max_t = df_all["time_idx"].max()
    train_cutoff = int(global_max_t - PRED_LEN)

    training = TimeSeriesDataSet(
        df_all[df_all.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx"],  # 필요시 요일/시각 피처 추가 가능
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

    model = DeepAR.from_dataset(
        training,
        learning_rate=LR_DEEPAR,
        hidden_size=HIDDEN,
        rnn_layers=RNN_LAYERS,
        dropout=DROPOUT,
        loss=NormalDistributionLoss()  # 평균/분산 학습 (예측 분포)
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

    # 최고 성능 ckpt를 고정명으로 복사
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        shutil.copyfile(best_ckpt, BEST_CKPT_PATH)
        shutil.copyfile(best_ckpt, LIVE_CKPT_PATH)
        print(f"✅ DeepAR 저장 완료\n - best: {best_ckpt}\n - BEST: {BEST_CKPT_PATH}\n - LIVE: {LIVE_CKPT_PATH}")
    else:
        # 예비: 현재 모델 저장
        fallback = os.path.join(DIR_MODELS, "multi_deepar_fallback.ckpt")
        trainer.save_checkpoint(fallback)
        shutil.copyfile(fallback, LIVE_CKPT_PATH)
        print(f"✅ DeepAR 수동 저장 완료: {fallback} & {LIVE_CKPT_PATH}")


# ====================== 실행부 ======================
if __name__ == "__main__":
    # 재현성
    seed_everything(42, workers=True)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True

    # 1) 데이터 수집 & 저장 (합본 + 심볼별 최신본)
    df_all = build_combined_csv()

    # 2) DeepAR 멀티시리즈 학습 (11개 입력/11개 예측)
    train_deepar_multi(df_all)

    print("🎯 ALL DONE")
