#!/usr/bin/env python3
import torch
import torch.nn as nn
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import time
import datetime
import os
import csv
from torch.cuda.amp import autocast
from torch.nn.utils import weight_norm
from train_regression_models import LSTMRegressor, TCNRegressor, predict_regression

# ─── 1) TCN 모델 정의 ─────────────────────────────────
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, d, p, drop):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p, dilation=d))
        self.chomp1 = Chomp1d(p); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(drop)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, k, stride=s, padding=p, dilation=d))
        self.chomp2 = Chomp1d(p); self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(drop)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()
    def forward(self, x):
        o = self.conv1(x); o = self.chomp1(o); o = self.relu1(o); o = self.drop1(o)
        o = self.conv2(o); o = self.chomp2(o); o = self.relu2(o); o = self.drop2(o)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(o + res)

class TCNBinaryClassifier(nn.Module):
    def __init__(self, num_inputs, channels=[32,64,128,256], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            in_ch = num_inputs if i == 0 else channels[i-1]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, ch, kernel_size, 1, dilation, padding, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(channels[-1], 1)
    def forward(self, x):
        x = x.permute(0,2,1)
        y = self.tcn(x)[:, :, -1]
        return self.fc(y)

# ─── 2) LSTM 모델 정의 ─────────────────────────────────
class BinaryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─── 3) 실시간 데이터 fetch ────────────────────────────
def get_latest_data(symbol: str, limit: int = 61):
    url    = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    data   = requests.get(url, params=params).json()
    df     = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    df     = df[["open","high","low","close"]].astype(float)
    df.columns = ["Open","High","Low","Close"]
    df.ta.ema(length=10, append=True)
    df.ta.rsi(length=14, append=True)
    df.dropna(inplace=True)
    return df

# ─── 4) 앙상블 예측 함수 (다수결) ─────────────────────────────
def predict_ensemble(tcn_model, lstm_model, scaler, df, thresh_tcn, thresh_lstm, device):
    seq = df[['High','Low','Open','Close','EMA_10','RSI_14']].values[-60:]
    x_scaled = scaler.transform(seq)
    xb = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    tcn_model.eval(); lstm_model.eval()
    with torch.no_grad(), autocast():
        logit_t = tcn_model(xb).squeeze().cpu().item()
        logit_l = lstm_model(xb).squeeze().cpu().item()
        prob_t  = torch.sigmoid(torch.tensor(logit_t)).item()
        prob_l  = torch.sigmoid(torch.tensor(logit_l)).item()

    # 각 모델 예측
    pred_t = 1 if prob_t >= thresh_tcn else 0
    pred_l = 1 if prob_l >= thresh_lstm else 0
    # (1) 평균 확률 vs 평균 임계치 방식
    avg_prob   = (prob_t + prob_l) / 2
    avg_thresh = (thresh_tcn + thresh_lstm) / 2
    pred       = 1 if avg_prob >= avg_thresh else 0

    return pred, avg_prob, prob_t, prob_l

# ─── 5) 메인 실행부 ─────────────────────────────────────────
if __name__ == "__main__":
    symbol       = "1000PEPEUSDC"
    tcn_path     = f"{symbol}_TCN_best.pth"
    lstm_path    = f"{symbol}_LSTM_best.pth"
    scaler_path  = f"{symbol}_binary_scaler.pkl"
    tcn_thresh_p = f"{symbol}_TCN_thresh.npy"
    lstm_thresh_p= f"{symbol}_LSTM_thresh.npy"
    csv_path     = "realtime_ensemble_results.csv"

    if not os.path.isfile(csv_path):
        with open(csv_path,'w',newline='') as f:
            csv.writer(f).writerow([
                'Timestamp','Predicted','EnsembleProb','TCNProb','LSTMProb','ActualChange'
            ])

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcn_model  = TCNBinaryClassifier(num_inputs=6).to(device)
    lstm_model = BinaryLSTM(input_dim=6).to(device)
    tcn_model.load_state_dict(torch.load(tcn_path, map_location=device))
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    scaler     = joblib.load(scaler_path)
    thresh_tcn = float(np.load(tcn_thresh_p))
    thresh_lstm= float(np.load(lstm_thresh_p))

    print(f"\n📡 실시간 앙상블 예측 시작 - {symbol}")
    print(f"   TCN thresh={thresh_tcn:.3f}, LSTM thresh={thresh_lstm:.3f}\n")

    # 예시: realtime_position_predictor.py 안에서

    # 임계치 정의
    # 예시: realtime_position_predictor.py 안에서
    reg_scaler = joblib.load('1000PEPEUSDC_reg_scaler.pkl')
    # load whichever regression you want (e.g. LSTMRegressor)
    reg_model = LSTMRegressor(input_dim=6).to(device)
    reg_model.load_state_dict(torch.load('LSTM_reg_best.pth', map_location=device))
    reg_model.eval()

    REG_THRESH = 0.25  # 회귀 필터 임계치 (%) 레버리지 20배 기준. 무레버리지면 0.08, 여유를 두고싶으면 5.0
    # **Classification thresholds**
    thr_tcn = 0.474
    thr_lstm = 0.479
    INVEST_AMT = 1000.0  # USDC 단위로 투자 금액

    # ─── Main Loop ────────────────────────────────────
    next_close = 0
    while True:
        # 1) 1분봉 데이터 불러오기
        df = get_latest_data(symbol)
        prev_close = df['Close'].iloc[-2]

        # 2) Ensemble 분류 예측
        use_df = df.iloc[:-1]
        pred_dir, avg_p, prob_t, prob_l = predict_ensemble(
                   tcn_model, lstm_model, scaler,
                   use_df, thr_tcn, thr_lstm, device)
        dir_str = "Long" if pred_dir == 1 else "Short"

        # 3) 회귀 예측 (% change → price)
        pred_pct   = predict_regression(reg_model, reg_scaler, df, device)
        pred_price = prev_close * (1 + pred_pct / 100)
        print(f"[DEBUG] prev_close={next_close:.7f}, pred_price={pred_price:.7f}")

        # 회귀 모델이 보는 Long/Short

        if pred_price > next_close: reg_pos = "Long"
        elif pred_price < next_close: reg_pos = "Short"
        else:                          reg_pos = "Flat"

        # 4) 예상 달러 손익 & ROI (투자금 1,000 USDC 기준)
        INVEST_AMT      = 1000.0
        pred_profit_usd = INVEST_AMT * pred_pct / 100
        pred_roi_pct    = pred_profit_usd / INVEST_AMT * 100

        # 5) 진입 필터 (기존 REG_THRESH 기준)
        do_trade = (
            (pred_dir == 1 and pred_pct >= REG_THRESH) or
            (pred_dir == 0 and pred_pct <= -REG_THRESH)
        )

        # 6) 예측 결과 출력
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{now}] ClsDir={dir_str}, ClsProb={avg_p:.2%}, "
            f"RegPos={reg_pos}, PredPrice={pred_price:.7f}, "
            f"PredPct={pred_pct:+.4f}%, "
            f"(PredProfit={pred_profit_usd:+.2f} USDC, ROI={pred_roi_pct:+.2f}%) → "
            f"{'✅' if do_trade else '⏳'}"
        )

        # 7) 1분 대기 후 실제 결과
        time.sleep(60)
        df2        = get_latest_data(symbol)
        next_close = df2['Close'].iloc[-1]

        # 실제 변화율 & 달러 손익 & ROI
        if round(next_close, 7) == round(prev_close, 7):
            actual_change = 0.0
        else:
            actual_change = (next_close - prev_close) / prev_close * 100
        actual_profit = INVEST_AMT * actual_change / 100
        actual_roi    = actual_profit / INVEST_AMT * 100
        # 가격 차이
        price_diff    = next_close - prev_close
        actual_pos = "Long" if price_diff > 0 else ("Short" if price_diff < 0 else "Flat")

        print(f"→ 실제 Close={next_close:.7f}, ΔPct={actual_change:+.4f}%, "
             f"(Profit={actual_profit:+.2f} USDC, ROI={actual_roi:+.2f}%), "
             f"ΔPrice={price_diff:+.7f}, ActualPos={actual_pos}\n")