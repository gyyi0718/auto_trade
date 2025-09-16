#!/usr/bin/env python3
import os, time, hmac, hashlib, requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta
from torch.cuda.amp import autocast
from torch.nn.utils import weight_norm
from train_regression_models import LSTMRegressor, TCNRegressor, predict_regression

# 1) 모델 정의 (실거래 스크립트와 동일)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size=chomp_size
    def forward(self,x): return x[:,:, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self,in_ch,out_ch,k,s,d,p,drop):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_ch,out_ch,k,stride=s,padding=p,dilation=d))
        self.chomp1=Chomp1d(p); self.relu1=nn.ReLU(); self.drop1=nn.Dropout(drop)
        self.conv2 = weight_norm(nn.Conv1d(out_ch,out_ch,k,stride=s,padding=p,dilation=d))
        self.chomp2=Chomp1d(p); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(drop)
        self.downsample = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else None
        self.final_relu = nn.ReLU()
    def forward(self,x):
        o=self.conv1(x); o=self.chomp1(o); o=self.relu1(o); o=self.drop1(o)
        o=self.conv2(o); o=self.chomp2(o); o=self.relu2(o); o=self.drop2(o)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(o+res)

class TCNBinaryClassifier(nn.Module):
    def __init__(self,num_inputs,channels=[32,64,128,256],k=3,drop=0.2):
        super().__init__()
        layers=[]
        for i,ch in enumerate(channels):
            in_ch = num_inputs if i==0 else channels[i-1]
            d=2**i; p=(k-1)*d
            layers.append( TemporalBlock(in_ch,ch,k,1,d,p,drop) )
        self.tcn=nn.Sequential(*layers)
        self.fc=nn.Linear(channels[-1],1)
    def forward(self,x):
        x=x.permute(0,2,1)
        y=self.tcn(x)[:,:, -1]
        return self.fc(y)

class BinaryLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim=64):
        super().__init__()
        self.lstm=nn.LSTM(input_dim,hidden_dim,batch_first=True)
        self.fc=nn.Linear(hidden_dim,1)
    def forward(self,x):
        out,_=self.lstm(x)
        return self.fc(out[:,-1,:])

# 2) Binance 1m klines 가져오는 함수
def fetch_klines(symbol, start_ts, end_ts, interval='1m'):
    """
    Fetch 1m klines from Binance futures between start_ts and end_ts (both in seconds).
    """
    url = 'https://fapi.binance.com/fapi/v1/klines'
    all_data = []
    limit = 1000
    t0 = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)

    while t0 < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': t0,
            'endTime': end_ms,
            'limit': limit
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # if Binance returned an error dict or something unexpected, stop
        if not isinstance(data, list):
            print("⚠️ Unexpected response from Binance:", data)
            break

        # if no more data, we're done
        if len(data) == 0:
            break

        all_data.extend(data)
        # advance to just after the last timestamp
        t0 = data[-1][0] + 1
        time.sleep(0.1)

    if not all_data:
        raise RuntimeError("No klines data fetched. Check symbol/date range.")

    df = pd.DataFrame(all_data, columns=[
        'open_time','o','h','l','c','v',
        'close_time','qav','trades','tbav','tbqv','ignore'
    ])
    df = df[['open_time','o','h','l','c']].astype(float)
    df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close'}, inplace=True)
    df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('dt', inplace=True)
    return df[['Open','High','Low','Close']]


# 3) 예측 함수
def predict_ensemble(tcn,lstm,scaler,window,th_t,th_l,device):
    seq = window[['High','Low','Open','Close','EMA_10','RSI_14']].values
    xb  = torch.tensor(scaler.transform(seq),dtype=torch.float32).unsqueeze(0).to(device)
    tcn.eval(); lstm.eval()
    with torch.no_grad(), autocast():
        pt = torch.sigmoid(tcn(xb).squeeze()).item()
        pl = torch.sigmoid(lstm(xb).squeeze()).item()
    pr = (pt+pl)/2
    dir = 1 if (pt>=th_t and pl>=th_l) else 0
    return dir, pr

# 4) 백테스트
def backtest(symbol, start, end, TP=0.25, SL=0.15):
    # 4.1) 데이터 로드
    df = fetch_klines(symbol, start.timestamp(), end.timestamp())
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)

    # 4.2) 모델·스케일러·임계치 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcn = TCNBinaryClassifier(6).to(device)
    lst = BinaryLSTM(6).to(device)
    tcn.load_state_dict(torch.load(f"{symbol}_TCN_best.pth",map_location=device))
    lst.load_state_dict(torch.load(f"{symbol}_LSTM_best.pth",map_location=device))
    scaler   = joblib.load(f"{symbol}_binary_scaler.pkl")
    th_t     = float(np.load(f"{symbol}_TCN_thresh.npy"))
    th_l     = float(np.load(f"{symbol}_LSTM_thresh.npy"))

    reg_s = joblib.load(f"{symbol}_reg_scaler.pkl")
    reg_m = LSTMRegressor(6).to(device)
    reg_m.load_state_dict(torch.load(f"{symbol}_LSTM_reg_best.pth",map_location=device))
    reg_m.eval()

    trades = []
    times = sorted(df.index)
    for i in range(60, len(times)-1):
        window = df.iloc[i-60:i]
        now    = times[i]
        close  = df.loc[now,'Close']

        # 1) 분류 예측
        dir, prob = predict_ensemble(tcn,lst,scaler,window,th_t,th_l,device)
        side = 'Long' if dir==1 else 'Short'

        # 2) 회귀 예측 (%)
        pct = predict_regression(reg_m, reg_s, df.iloc[i-60:i+1], device)
        # 3) 진입 필터
        if not ((dir==1 and pct>=TP) or (dir==0 and pct<=-TP)):
            continue

        entry_price = close
        entry_time  = now

        # 4) TP/SL 모니터링 (다음 60캔들 혹은 조건 만족)
        exit_price, exit_time = entry_price, entry_time
        for j in range(i+1, min(i+61, len(times))):
            t2 = times[j]
            price = df.loc[t2,'Close']
            change = (price-entry_price)/entry_price*100 * (1 if dir==1 else -1)
            if change>=TP or change<=-SL:
                exit_price, exit_time = price, t2
                break
            exit_price, exit_time = price, t2

        pnl_usd = (exit_price-entry_price) * (1 if dir==1 else -1)
        roi     = pnl_usd/entry_price*100

        trades.append({
            'entry_time': entry_time, 'exit_time': exit_time,
            'side':side, 'entry':entry_price, 'exit':exit_price,
            'pnl': pnl_usd, 'ROI_pct': roi
        })

    # 5) 결과 정리
    df_tr = pd.DataFrame(trades)
    total_pnl = df_tr['pnl'].sum()
    win_rate  = (df_tr['pnl']>0).mean()
    avg_pnl   = df_tr['pnl'].mean()
    print("=== Backtest Results ===")
    print(f"Trades: {len(df_tr)}, Total PnL: {total_pnl:.4f}, Win rate: {win_rate:.2%}")
    print(f"Avg PnL/trade: {avg_pnl:.4f}")
    return df_tr

if __name__=='__main__':
    # 백테스트 기간 설정
    start = datetime(2025,1,1)
    end   = datetime(2025,7,1)
    df_trades = backtest("1000PEPEUSDC", start, end)
    # CSV로 저장
    df_trades.to_csv("backtest_results.csv", index=False)
