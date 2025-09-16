import os
import time
import torch
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
import warnings
from collections import deque

warnings.filterwarnings("ignore")

# ───── 설정 ─────
symbol = "DOGEUSDT"
seq_len = 60
pred_len = 10
interval = "1m"
thresh_roi = 0.001
slippage = 0.0005
fee_rate = 0.0004
take_profit_roi = 0.01
stop_loss_roi = -0.01
initial_capital = 1000.0
capital = initial_capital
SLEEP_SEC = 60
LOG_FILE = f"{symbol}_trade_log.csv"
SELECTION_MODE = "confidence"  # 선택: "confidence" 또는 "roi_ratio"



# ───── 로그 파일 초기화 ─────
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "event", "direction", "price", "roi", "value", "capital", "note"]).to_csv(LOG_FILE, index=False)

def log_event(event, direction, price, roi, value, capital, note=""):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "direction": direction,
        "price": price,
        "roi": roi,
        "value": value,
        "capital": capital,
        "note": note
    }
    pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode="a", header=False, index=False)

# ───── 모델 로딩 ─────
df_all = pd.read_csv(f"{symbol}_deepar_input.csv").dropna()
deepar_dataset = TimeSeriesDataSet(
    df_all,
    time_idx="time_idx",
    target="log_return",
    group_ids=["series_id"],
    max_encoder_length=seq_len,
    max_prediction_length=pred_len,
    time_varying_known_reals=["time_idx", "volume"],
    time_varying_unknown_reals=["log_return"],
    static_categoricals=[],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

model = DeepAR.load_from_checkpoint(f"{symbol}_deepar_model.ckpt")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ───── 함수 정의 ─────
def fetch_ohlcv(symbol, limit=2):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params).json()
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(res, columns=columns)
        df = df.astype({"close": float, "volume": float})
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] OHLCV fetch 실패: {e}")
        return pd.DataFrame()

def predict_with_uncertainty(df, now_price):
    df = df.copy().reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    df["series_id"] = symbol
    ds = TimeSeriesDataSet.from_dataset(deepar_dataset, df, predict=True, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=1)

    raw = model.predict(dl, mode="raw")[0]
    log_returns_all = raw.cpu().numpy()[0]
    log_returns_all = log_returns_all[:, :pred_len]

    pct_returns_all = np.exp(log_returns_all) - 1
    cum_returns_all = np.cumsum(np.log1p(pct_returns_all), axis=1)
    pred_prices_all = now_price * np.exp(cum_returns_all)

    avg_prices = np.mean(pred_prices_all, axis=0)
    lower_band = np.percentile(pred_prices_all, 10, axis=0)
    upper_band = np.percentile(pred_prices_all, 90, axis=0)
    avg_roi = np.mean(pct_returns_all, axis=0)

    return avg_prices, lower_band, upper_band, avg_roi

# ───── 상태 변수 초기화 ─────
is_holding = False
entry_price = None
entry_time = None
direction = None
target_hold_seconds = None
target_price = None

# ───── 실시간 루프 시작 전 데이터 초기화 ─────
data_queue = deque(maxlen=seq_len + pred_len)
print("[INFO] 초기 데이터 수집 중...")
while True:
    df_init = fetch_ohlcv(symbol, limit=seq_len + pred_len + 10)
    if not df_init.empty and df_init.shape[0] >= seq_len + pred_len:
        for _, row in df_init.tail(seq_len + pred_len).iterrows():
            data_queue.append(row)
        break
    print("[WAIT] 데이터 부족, 재시도 중...")
    time.sleep(3)
print("[INFO] 초기화 완료. 실시간 예측 시작")

# ───── 실시간 루프 ─────
# ───── 실시간 루프 ─────
while True:
    df_new = fetch_ohlcv(symbol, limit=2)
    if df_new.empty:
        print("[WARN] 최신 데이터 없음")
        if not is_holding:
            time.sleep(SLEEP_SEC)
        continue

    data_queue.append(df_new.iloc[-1])
    df_seq = pd.DataFrame(list(data_queue))

    if len(df_seq) < seq_len + pred_len:
        print(f"[{datetime.now()}] ⏳ 데이터 부족")
        if not is_holding:
            time.sleep(SLEEP_SEC)
        continue

    if not is_holding:
        now_price = df_seq["close"].iloc[-pred_len - 1]
        avg_prices, lower_band, upper_band, avg_roi_seq = predict_with_uncertainty(df_seq, now_price)

        # 🎯 가장 ROI 큰 시점 기준
        target_idx = np.argmax(np.abs(avg_roi_seq))
        target_roi = avg_roi_seq[target_idx]
        target_price = avg_prices[target_idx]
        target_min = target_idx + 1

        print("=" * 80)
        print(f"[{datetime.now()}] 🔮 Predicted ROI Sequence: {[f'{x*100:.4f}%' for x in avg_roi_seq]}")
        print(f"🎯 Target ROI = {target_roi * 100:.4f}% @ {target_min}분 후 → 예상 가격: {target_price:.6f} | 현재가: {now_price:.6f}")
        print("=" * 80)

        if abs(target_roi) > thresh_roi:
            entry_price = now_price * (1 + slippage)
            entry_time = datetime.now()
            direction = "LONG" if target_roi > 0 else "SHORT"
            is_holding = True
            target_hold_seconds = target_min * 60

            print(f"🚀 ENTRY [{direction}] @ {entry_price:.6f} | 청산 목표 {target_min}분 뒤 | 예상 ROI {target_roi*100:.3f}%")
            log_event("ENTRY", direction, entry_price, 0, capital, capital, f"target_min={target_min} ROI={target_roi:.5f}")

        time.sleep(SLEEP_SEC)

    else:
        # 실시간 청산 감시
        latest_price = df_new["close"].iloc[-1]
        roi = (latest_price - entry_price) / entry_price if direction == "LONG" else (entry_price - latest_price) / entry_price
        value = capital * (1 + roi - fee_rate)
        held_seconds = (datetime.now() - entry_time).seconds

        print(f"⏳ MONITOR [{direction}] @ {latest_price:.6f} ROI={roi*100:.4f}%  Value=${value:.4f} Held={held_seconds}s")

        if held_seconds < target_hold_seconds:
            # 🎯 예상 ROI 도달 시 조기 청산
            if (direction == "LONG" and roi >= target_roi) or (direction == "SHORT" and roi >= abs(target_roi)):
                print(f"💰 EARLY EXIT by TARGET ROI [{direction}] @ {latest_price:.6f} ROI={roi*100:.4f}%")
                log_event("EXIT_EARLY", direction, latest_price, roi, value, capital)
                capital = value
                is_holding = False
        else:
            # ⌛ 시간이 지났으면 TP/SL로 청산
            if roi >= take_profit_roi:
                print(f"💰 TAKE PROFIT [{direction}] @ {latest_price:.6f} Gain=${value - capital:.2f}")
                log_event("EXIT_TP", direction, latest_price, roi, value, capital)
                capital = value
                is_holding = False
            elif roi <= stop_loss_roi:
                print(f"🛑 STOP LOSS [{direction}] @ {latest_price:.6f} Loss=${value - capital:.2f}")
                log_event("EXIT_SL", direction, latest_price, roi, value, capital)
                capital = value
                is_holding = False
            else:
                print(f"⏱ HOLDING after target duration... TP/SL 조건 대기 중")
