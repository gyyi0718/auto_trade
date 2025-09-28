# ✅ 목적: 실시간 가격을 받아와 DeepAR 모델로 예측 후 Long/Short 진입 및 수익률 계산

import pandas as pd
import numpy as np
import torch
import time
import requests
from pytorch_forecasting import DeepAR, TimeSeriesDataSet

# === 설정 ===
SEQ_LEN = 60
PRED_LEN = 5
SYMBOL = "XRPUSDT"
INTERVAL = "1m"
FEE_RATE = 0.0004  # 수수료
SLIPPAGE = 0.0005  # 슬리피지
ENTRY_THRESHOLD = 0.002  # log return 기준 진입 임계값
API_URL = "https://fapi.binance.com/fapi/v1/klines"

# === 모델 로드 ===
model = DeepAR.load_from_checkpoint("deepar_model.ckpt")
model.eval()

# === 실시간 가격 수집 함수 ===
def fetch_latest_klines(symbol, interval, limit=100):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(API_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Number_of_trades",
        "Taker_buy_base_asset_volume", "Taker_buy_quote_asset_volume", "Ignore"
    ])
    df = df[["Timestamp", "Close", "Volume"]]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.astype({"Close": float, "Volume": float})
    return df.reset_index(drop=True)

# === 실시간 트레이딩 루프 ===
capital = 1000.0
position = None
entry_price = 0.0

print("🚀 실시간 DeepAR 예측 시작...")
while True:
    df = fetch_latest_klines(SYMBOL, INTERVAL, limit=SEQ_LEN + 1)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = SYMBOL

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series_id"],
        max_encoder_length=SEQ_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=["time_idx", "Volume"],
        time_varying_unknown_reals=["log_return"],
        static_categoricals=[],
        target_normalizer=model.dataset_parameters["target_normalizer"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    loader = dataset.to_dataloader(train=False, batch_size=1)
    predictions, _ = model.predict(loader, mode="raw", return_x=True)

    mu = predictions[0].output.prediction.mean.detach().cpu().numpy()
    pred_log_return = mu[-1]
    current_price = df["Close"].iloc[-1]

    # 진입 조건
    if position is None:
        if pred_log_return > ENTRY_THRESHOLD:
            position = "long"
            entry_price = current_price * (1 + SLIPPAGE)
            print(f"[LONG 진입] 가격: {entry_price:.5f}")
        elif pred_log_return < -ENTRY_THRESHOLD:
            position = "short"
            entry_price = current_price * (1 - SLIPPAGE)
            print(f"[SHORT 진입] 가격: {entry_price:.5f}")
    else:
        pnl = 0
        if position == "long":
            pnl = (current_price - entry_price) / entry_price - FEE_RATE
        elif position == "short":
            pnl = (entry_price - current_price) / entry_price - FEE_RATE
        capital *= (1 + pnl)
        print(f"[청산] 방향: {position.upper()}, 수익률: {pnl:.4f}, 자본: ${capital:.2f}")
        position = None
        entry_price = 0.0

    time.sleep(60)  # 1분 간격
