import os, time, warnings
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from collections import deque
import requests
from binance.client import Client
from decimal import Decimal, ROUND_DOWN

warnings.filterwarnings("ignore")

# ───── Config 설정 ─────
CONFIG = {
    "API_KEY": "YOUR_API_KEY",
    "API_SECRET": "YOUR_API_SECRET",
    "symbol": "1000PEPEUSDT",
    "seq_len": 60,
    "pred_len": 10,
    "interval": "1m",
    "SLEEP_SEC": 5,
    "slippage": 0.0005,
    "fee_rate": 0.0004,
    "thresh_roi": 0.0003,
    "ENTRY_WAIT": 5,
    "EXIT_WAIT": 3,
    "LEVERAGE": 20,
    "MIN_PROFIT_USD": 0.01,
    "ENTRY_PORTION": 0.2,
    "MAX_POSITION_VALUE": 1000,
    "fee_rate_ratio": 1.2,
    "min_notional": 5.0,
}

# ───── Binance API Wrapper ─────
class BinanceClientWrapper:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_balance(self, asset="USDT"):
        balances = self.client.futures_account_balance(recvWindow=10000)
        for b in balances:
            if b["asset"] == asset:
                return float(b["availableBalance"])
        return 0.0

    def place_market_order(self, symbol, side, qty):
        return self.client.futures_create_order(symbol=symbol, side=side, type="MARKET", quantity=qty, recvWindow=10000)

    def place_limit_order(self, symbol, side, qty, price):
        return self.client.futures_create_order(symbol=symbol, side=side, type="LIMIT", timeInForce="GTC", quantity=qty, price=round(price, 6), recvWindow=10000)

    def wait_until_filled(self, order_id, symbol, timeout):
        start = time.time()
        while time.time() - start < timeout:
            status = self.client.futures_get_order(symbol=symbol, orderId=order_id)
            if status["status"] == "FILLED":
                return True
            time.sleep(1)
        return False

# ───── Logger ─────
class Logger:
    def __init__(self, symbol):
        self.log_file = f"{symbol}_trade_log.csv"
        self.pred_file = f"{symbol}_predict_log.csv"
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=["timestamp", "event", "direction", "price", "roi", "value", "capital", "note"]).to_csv(self.log_file, index=False)
        if not os.path.exists(self.pred_file):
            pd.DataFrame(columns=["timestamp", "now_price", "target_price", "pred_pct_roi", "target_min", "target_roi", "real_roi", "entered"]).to_csv(self.pred_file, index=False)

    def log_event(self, event, direction, price, roi, value, capital, note=""):
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "event": event, "direction": direction, "price": price, "roi": roi, "value": value, "capital": capital, "note": note}
        pd.DataFrame([row]).to_csv(self.log_file, mode="a", header=False, index=False)

    def log_prediction(self, now_price, target_price, target_min, target_roi, real_roi, entered):
        pred_pct_roi = ((target_price / now_price) - 1) * 100
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "now_price": now_price, "target_price": target_price, "pred_pct_roi": pred_pct_roi, "target_min": target_min, "target_roi": target_roi, "real_roi": real_roi, "entered": entered}
        pd.DataFrame([row]).to_csv(self.pred_file, mode="a", header=False, index=False)

# ───── Predictor ─────
class Predictor:
    def __init__(self, config):
        self.symbol = config["symbol"]
        df_all = pd.read_csv(f"{self.symbol}_deepar_input.csv").dropna()
        self.dataset = TimeSeriesDataSet(df_all,
            time_idx="time_idx", target="log_return", group_ids=["series_id"],
            max_encoder_length=config["seq_len"], max_prediction_length=config["pred_len"],
            time_varying_known_reals=["time_idx", "volume"], time_varying_unknown_reals=["log_return"],
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True)
        self.model = DeepAR.load_from_checkpoint(f"{self.symbol}_deepar_model.ckpt")
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def fetch_ohlcv(self, symbol, interval, limit):
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        res = requests.get(url, params=params).json()
        df = pd.DataFrame(res, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "q", "n", "tb", "tq", "ignore"])
        df = df.astype({"close": float, "volume": float})
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df.dropna()

    def predict(self, df, now_price, config):
        df = df.copy().reset_index(drop=True)
        df["time_idx"] = np.arange(len(df))
        df["series_id"] = config["symbol"]
        ds = TimeSeriesDataSet.from_dataset(self.dataset, df, predict=True, stop_randomization=True)
        dl = ds.to_dataloader(train=False, batch_size=1)
        raw = self.model.predict(dl, mode="raw")[0].cpu().numpy()[0, :, :config["pred_len"]]
        pct_returns = np.exp(raw) - 1
        avg_roi = np.mean(pct_returns, axis=0)
        cum = np.cumsum(np.log1p(pct_returns), axis=1)
        pred_prices = now_price * np.exp(cum)
        return np.mean(pred_prices, axis=0), avg_roi
