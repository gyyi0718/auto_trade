import os, traceback
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer

# ===== 사용자 설정 =====
SEQ_LEN    = 60
PRED_LEN   = 60
INPUT_CSV  = "data_upbit/upbit_top10_1m.csv"
CKPT_PATH  = "models/upbit_multi_deepar_model.ckpt"
OUTPUT_CSV = "upbit_output.csv"

# ===== 데이터 로드 & 전처리 =====
df = pd.read_csv(INPUT_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
df["close"] = pd.to_numeric(df["close"], errors="coerce")

# time_idx 부여
df["time_idx"] = df.groupby("series_id").cumcount()

# log_return 생성
df["log_return"] = df.groupby("series_id")["close"].transform(lambda s: np.log(s).diff())
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["log_return", "close", "timestamp"]).reset_index(drop=True)

print("✅ 칼럼:", df.columns.tolist())
print("✅ 샘플:", df[["series_id", "time_idx", "close", "log_return"]].tail())

# ===== 모델 로드 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DeepAR.load_from_checkpoint(CKPT_PATH, map_location=device)
model.eval()

# ===== 전체 구조 정의 (from_dataset 기준용) =====

full_dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="log_return",
    group_ids=["series_id"],
    max_encoder_length=SEQ_LEN,
    max_prediction_length=PRED_LEN,
    static_categoricals=["series_id"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["log_return"],
    target_normalizer=GroupNormalizer(groups=["series_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)


# ===== 예측 수행 =====
pred_rows = []
symbols = df["series_id"].unique().tolist()

for symbol in symbols:
    try:
        d = df[df["series_id"] == symbol].copy()
        if len(d) < SEQ_LEN + 1:
            print(f"[SKIP] {symbol}: history too short ({len(d)})")
            continue

        max_t = d["time_idx"].max()
        enc = d[d["time_idx"] > max_t - SEQ_LEN].copy()

        center = float(enc["log_return"].mean())
        scale  = max(1e-6, float(enc["log_return"].std()))

        # 디코더(미래) 프레임 — log_return은 반드시 넣어야 함
        dec = pd.DataFrame({
            "series_id": symbol,
            "time_idx": list(range(max_t + 1, max_t + PRED_LEN + 1)),
            "close": enc["close"].iloc[-1],  # just for info
            "log_return": [0.0] * PRED_LEN  # ✅ 핵심 포인트
        })

        inp = pd.concat([enc, dec], ignore_index=True)

        pred_ds = TimeSeriesDataSet.from_dataset(
            full_dataset, inp, predict=True, stop_randomization=True
        )
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        with torch.no_grad():
            out = model.predict(pred_loader, return_index=True, return_y=False)

        preds = out.output[0]
        idxs  = out.index["time_idx"].tolist()

        for t, v in zip(idxs, preds):
            pred_rows.append({
                "series_id": symbol,
                "time_idx":  int(t),
                "pred_log_return": float(v),
            })

        print(f"[OK] {symbol}: {len(preds)} steps")

    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        traceback.print_exc()

# ===== 저장 =====
out_df = pd.DataFrame(pred_rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 예측 완료: {len(out_df)}개 저장 → {OUTPUT_CSV}")
