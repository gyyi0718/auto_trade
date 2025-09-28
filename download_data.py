import pandas as pd
import requests
import time
import pandas_ta as ta
from datetime import datetime, timedelta
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from train_regression_models import LSTMRegressor, TCNRegressor, predict_regression
from train_binary_models import  TCNBinaryClassifier,BinaryLSTM

def download_binance_1m_range(symbol: str, start_date: str, end_date: str, save_path: str):
    all_data = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while start <= end:
        start_dt = datetime.combine(start, datetime.min.time()) + timedelta(hours=8)
        end_dt   = datetime.combine(start, datetime.min.time()) + timedelta(hours=16)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts   = int(end_dt.timestamp() * 1000)

        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }

        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
        ])

        if not df.empty:
            df = df[["open", "high", "low", "close"]].astype(float)
            df.columns = ["Open", "High", "Low", "Close"]
            all_data.append(df)

        start += timedelta(days=1)
        time.sleep(0.3)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.ta.ema(length=10, append=True)
    final_df.ta.rsi(length=14, append=True)
    final_df.dropna(inplace=True)
    final_df.to_csv(save_path, index=False)
    print(f"✅ 저장 완료: {save_path} | 총 {len(final_df)}개 캔들")

def run_backtest_with_df(df, symbol="1000PEPEUSDC", initial_capital=1000.0, roi_threshold=0.001, fee_rate=0.0004, slippage_rate=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 및 스케일러 로드
    tcn_model = TCNBinaryClassifier(input_dim=6).to(device)
    tcn_model.load_state_dict(torch.load(f"{symbol}_TCN_best.pth", map_location=device))

    lstm_model = BinaryLSTM(input_dim=6).to(device)
    lstm_model.load_state_dict(torch.load(f"{symbol}_LSTM_best.pth", map_location=device))

    reg_model = LSTMRegressor(input_dim=6).to(device)
    reg_model.load_state_dict(torch.load(f"{symbol}_LSTM_reg_best.pth", map_location=device))
    reg_model.eval()

    scaler = joblib.load(f"{symbol}_binary_scaler.pkl")
    reg_scaler = joblib.load(f"{symbol}_reg_scaler.pkl")
    thr_tcn = float(np.load(f"{symbol}_TCN_thresh.npy"))
    thr_lstm = float(np.load(f"{symbol}_LSTM_thresh.npy"))

    def predict_direction(hist):
        seq = hist[['High','Low','Open','Close','EMA_10','RSI_14']].values[-60:]
        x_scaled = scaler.transform(seq)
        xb = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logit_t = tcn_model(xb).squeeze().cpu().item()
            logit_l = lstm_model(xb).squeeze().cpu().item()
            prob_t = torch.sigmoid(torch.tensor(logit_t)).item()
            prob_l = torch.sigmoid(torch.tensor(logit_l)).item()
        avg_prob = (prob_t + prob_l) / 2
        avg_thresh = (thr_tcn + thr_lstm) / 2
        return 1 if avg_prob >= avg_thresh else 0, avg_prob

    capital = initial_capital
    trades = []

    for i in range(61, len(df) - 1):
        hist = df.iloc[i-60:i]
        now_price = df['Close'].iloc[i]
        next_price = df['Close'].iloc[i+1]

        direction, prob = predict_direction(hist)
        reg_pct = predict_regression(reg_model, reg_scaler, hist, device)
        pred_price = now_price * (1 + reg_pct / 100)

        if abs(reg_pct / 100) < roi_threshold:
            trades.append({
                "Index": i,
                "NowPrice": now_price,
                "NextPrice": next_price,
                "Direction": "SKIP",
                "ClsProb": prob,
                "PredPrice": pred_price,
                "ExpectedROI": reg_pct / 100,
                "ActualROI": 0,
                "TradeROI": 0,
                "Capital": capital
            })
            continue

        actual_roi = (next_price - now_price) / now_price
        trade_roi = actual_roi if direction == 1 else -actual_roi
        trade_roi -= (fee_rate * 2 + slippage_rate)
        capital *= (1 + trade_roi)

        trades.append({
            "Index": i,
            "NowPrice": now_price,
            "NextPrice": next_price,
            "Direction": "Long" if direction == 1 else "Short",
            "ClsProb": prob,
            "PredPrice": pred_price,
            "ExpectedROI": reg_pct / 100,
            "ActualROI": actual_roi,
            "TradeROI": trade_roi,
            "Capital": capital
        })

    result_df = pd.DataFrame(trades)
    result_df.to_csv("backtest_result.csv", index=False)

    plt.figure(figsize=(12,6))
    plt.plot(result_df['Capital'], label='Capital')
    plt.title('Backtest Capital Growth')
    plt.xlabel('Trades')
    plt.ylabel('Capital (USDC)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest_capital_chart.png")
    plt.close()

    print(result_df.tail())
    print(f"\n💰 Final Capital: ${capital:.2f}")
    print(f"📊 Total ROI: {((capital - initial_capital) / initial_capital) * 100:.2f}%")
    print("📁 Results saved to 'backtest_result.csv' and 'backtest_capital_chart.png'")
    return result_df

def load_data_and_backtest(csv_path):
    df = pd.read_csv(csv_path)
    return run_backtest_with_df(df)

# 예시 사용 방법
if __name__ == "__main__":
    SYMBOL = "1000PEPEUSDC"
    START_DATE = "2025-07-24"
    END_DATE = "2025-07-30"
    SAVE_PATH = f"{SYMBOL}_{START_DATE}_to_{END_DATE}.csv"

    # 1) 데이터 다운로드 및 저장
    download_binance_1m_range(SYMBOL, START_DATE, END_DATE, SAVE_PATH)

    # 2) 저장된 데이터를 이용한 백테스트 실행
    load_data_and_backtest(SAVE_PATH)
