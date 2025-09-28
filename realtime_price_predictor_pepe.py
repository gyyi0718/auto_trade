import torch
import torch.nn as nn
import requests
import pandas as pd
import numpy as np
import time
import datetime
import pandas_ta as ta
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# --- Model Definition ---
class TransformerPricePredictor(nn.Module):
    def __init__(self, seq_len, input_dim=6, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]
        x = self.output_proj(x)
        return x

# --- Utility to get Binance minute kline data ---
def get_latest_minute_data(symbol: str, interval: str = "1m", limit: int = 61):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
    df = df[["open", "high", "low", "close"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close"]
    df.ta.ema(length=10, append=True)
    df.ta.rsi(length=14, append=True)
    df.dropna(inplace=True)
    return df

# --- Real-time Prediction ---
def predict_next_minute(model, scaler, symbol="1000PEPEUSDC"):
    df = get_latest_minute_data(symbol)
    input_data = df[['High', 'Low', 'Open', 'Close', 'EMA_10', 'RSI_14']].values[-61:]
    scaled_input = scaler.transform(input_data[-61:-1])  # use last 60 for input
    actual_next = input_data[-1]  # actual current minute (used as label)

    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).numpy()
    pred_inverse = scaler.inverse_transform(prediction)[0]  # shape: (6,)
    return pred_inverse, actual_next

if __name__ == "__main__":
    symbol = "1000PEPEUSDC"
    model_path = "transformer_price_model.pth"
    scaler_path = "1000PEPEUSDC_scaler_250727.pkl"
    result_csv = f"{symbol}_live_prediction_log.csv"

    # Load model and scaler
    model = TransformerPricePredictor(seq_len=60, input_dim=6)
    model.load_state_dict(torch.load(model_path))
    scaler = joblib.load(scaler_path)

    print("📡 Real-Time 1-minute Price Prediction for", symbol)

    while True:
        try:
            pred, actual = predict_next_minute(model, scaler, symbol)
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pred_close = pred[3]
            actual_close = actual[3]
            diff = actual_close - pred_close

            print(f"[{now}] Predicted Close: {pred_close:.8f}, Actual Close: {actual_close:.8f}, Diff: {diff:.8f}")

            # Save to CSV
            result = pd.DataFrame([[now, pred_close, actual_close, diff]],
                                  columns=["time", "predicted_close", "actual_close", "diff"])

            if not os.path.isfile(result_csv):
                result.to_csv(result_csv, mode='w', index=False)
            else:
                result.to_csv(result_csv, mode='a', header=False, index=False)

            time.sleep(60)
        except Exception as e:
            print("❌ Error during prediction:", e)
            time.sleep(60)
            continue
