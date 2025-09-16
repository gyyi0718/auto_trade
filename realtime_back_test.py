import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import time
import datetime
from torch.cuda.amp import autocast
from train_regression_models import TCNRegressor
from sklearn.preprocessing import MinMaxScaler
import requests
import pandas_ta as ta

# ───── 설정 ─────
symbol = "XRPUSDT"
seq_len = 60
SLEEP_SEC = 60
INITIAL_CAPITAL = 1000.0
TRADE_AMOUNT = 1000.0  # 투자 단위 자산
TRAILING_GAP = 0.3  # 수익 최대값 대비 감소폭 (달러 기준)
MAX_HOLD_SECONDS = 300  # 최대 보유시간 5분
FEE_RATE = 0.0004
SLIPPAGE = 0.0005
# 상단 파라미터 설정 추가
RECOVERY_MIN_PROFIT = 0.002     # +0.2% 이상 수익일 때만 회복 청산
RECOVERY_TRAIL_GAP = 0.1        # 회복 후 피크 대비 0.1달러 하락 시 청산
TRAILING_PCT = 0.3  # 최고 수익 대비 30% 하락 시 청산

LOG_FILE = "trade_log.csv"

# ✔ 익절/손절 기준 ROI (% 단위 아님)
TAKE_PROFIT_ROI = 0.007   # +0.7% 익절
STOP_LOSS_ROI = -0.005    # -0.5% 손절

capital = INITIAL_CAPITAL
is_holding = False
entry_price = None
entry_time = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───── 모델 및 스케일러 로드 ─────
tcn_path = f"./checkpoints/TCN_pctchange_best.pth"
scaler_path = f"{symbol}_pct_scaler.pkl"

model = TCNRegressor(input_dim=6).to(device)
model.load_state_dict(torch.load(tcn_path, map_location=device))
model.eval()
scaler = joblib.load(scaler_path)

def get_adaptive_gap(peak_profit):
    if peak_profit < 0.5:
        return 0.1
    elif peak_profit < 2.0:
        return 0.3
    else:
        return 0.5
# ───── 실시간 데이터 불러오기 함수 ─────
def fetch_latest_ohlcv(symbol: str, limit: int = 100):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    df = df[["open", "high", "low", "close"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close"]
    df.ta.ema(length=10, append=True)
    df.ta.rsi(length=14, append=True)
    df.dropna(inplace=True)
    return df

# ───── 예측 함수 ─────
def predict_pct_change(model, scaler, df):
    seq = df[['High','Low','Open','Close','EMA_10','RSI_14']].values[-seq_len:]
    if np.isnan(seq).any():
        return float('nan')
    x_scaled = scaler.transform(seq)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad(), autocast():
        pct = model(x_tensor).squeeze().cpu().item()
    return pct

# ───── 트레이드 로그 저장 ─────
def log_trade(log_dict):
    df_log = pd.DataFrame([log_dict])
    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)

# ───── 메인 루프 ─────
print(f"\n📡 실시간 회귀 기반 트레이딩 시작 - {symbol} (트레일링+고정익절/손절 모드)")

while True:
    now_dt = datetime.datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    df = fetch_latest_ohlcv(symbol, limit=100)
    if df.shape[0] < seq_len:
        print(f"[{now_str}] ⚠ 데이터 부족 ({df.shape[0]}/{seq_len})")
        time.sleep(SLEEP_SEC)
        continue

    prev_close = df['Close'].iloc[-2]
    latest_close = df['Close'].iloc[-1]

    pred_pct = predict_pct_change(model, scaler, df)
    if np.isnan(pred_pct):
        print(f"[{now_str}] ⚠ NaN 예측 → 스킵")
        time.sleep(SLEEP_SEC)
        continue

    pred_price = prev_close * (1 + pred_pct / 100)
    delta_price = pred_price - prev_close
    delta_pct = pred_pct

    print(f"[{now_str}] 🎯 Pred={pred_price:.7f}  Prev={prev_close:.7f}  ΔPrice={delta_price:+.7f}  ROI={delta_pct:+.5f}%  Capital={capital:.2f} USDC")

    if not is_holding:
        is_holding = True
        entry_price = prev_close
        entry_time = datetime.datetime.now()
        FEE_SLIPPAGE = (FEE_RATE * 2 + SLIPPAGE) * TRADE_AMOUNT
        min_profit = 0
        peak_profit = 0
        exited = False

        print(f"[{now_str}] 🚀 ENTRY @ {entry_price:.7f}  Pred={pred_price:.7f}  ROI={delta_pct:+.5f}%")

        while True:
            time.sleep(1)
            now_dt = datetime.datetime.now()
            now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            df_live = fetch_latest_ohlcv(symbol, limit=2)
            curr_price = df_live['Close'].iloc[-1]
            held_secs = (now_dt - entry_time).total_seconds()

            price_diff = curr_price - entry_price
            gross_profit = price_diff * TRADE_AMOUNT / entry_price
            net_profit = gross_profit - FEE_SLIPPAGE
            roi = net_profit / TRADE_AMOUNT

            if net_profit < min_profit:
                min_profit = net_profit
            if net_profit > peak_profit:
                peak_profit = net_profit

            print(f"[{now_str}] ⏳ MONITOR Entry={entry_price:.7f} Now={curr_price:.7f} Profit=${net_profit:+.4f} ROI={roi*100:+.2f}% Held={held_secs:.0f}s")

            # 기존 청산 조건 대체
            if roi >= TAKE_PROFIT_ROI:
                result = "FIXED_TAKE_PROFIT"
                exited = True
            elif roi <= STOP_LOSS_ROI:
                result = "FIXED_STOP_LOSS"
                exited = True
            elif min_profit < 0 and roi >= RECOVERY_MIN_PROFIT:
                result = "RECOVERY_MIN"
                exited = True
            elif min_profit < 0 and net_profit >= 0 and (peak_profit - net_profit) >= RECOVERY_TRAIL_GAP:
                result = "RECOVERY_TRAILING"
                exited = True
            elif net_profit >= 0 and (peak_profit - net_profit) >= TRAILING_GAP:
                result = "TRAILING_EXIT"
                exited = True
            elif held_secs >= MAX_HOLD_SECONDS and net_profit >= 0:
                result = "TIMEOUT"
                exited = True
            # 청산 조건부 내에 추가
            elif peak_profit > 0 and (peak_profit - net_profit) / peak_profit >= TRAILING_PCT:
                result = "TRAILING_PCT_EXIT"
                exited = True
            elif net_profit >= 0 and (peak_profit - net_profit) >= get_adaptive_gap(peak_profit):
                result = "ADAPTIVE_TRAILING_EXIT"
                exited = True

            if exited:
                capital += net_profit
                print(f"[{now_str}] 💰 EXIT {result} @ {curr_price:.7f} NetProfit=${net_profit:+.4f} Capital={capital:.2f} USDC")
                log_trade({
                    "Time": now_str,
                    "EntryPrice": entry_price,
                    "ExitPrice": curr_price,
                    "ProfitUSD": net_profit,
                    "Capital": capital,
                    "Result": result
                })
                is_holding = False
                break

    time.sleep(SLEEP_SEC)
