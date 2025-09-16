#!/usr/bin/env python3
import os, time, hmac, hashlib, json
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import joblib
import pandas_ta as ta
from datetime import datetime
from torch.nn.utils import weight_norm

# ─── 1) 환경변수에서 API 키 읽기 ─────────────────────────
API_KEY    = "kynaVBENI8TIS8AcZICK2Sn52mJfIqHpCHfKv2IaSM7TQtKaazn44KGcsJRERjbE"
API_SECRET = "z2ZtL0S8eIE34C8bL1XpAV0jEhcc8JEm8VBOEZbo5Rbe4d9HkN2F4V2AGukhtXyT"

# ─── 2) 모델 정의(학습 스크립트와 동일) ─────────────────────
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size=chomp_size
    def forward(self,x): return x[:,:, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self,in_ch,out_ch,k,s,d,p,drop):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_ch,out_ch,k,stride=s,padding=p,dilation=d))
        self.chomp1 = Chomp1d(p); self.relu1=nn.ReLU(); self.drop1=nn.Dropout(drop)
        self.conv2 = weight_norm(nn.Conv1d(out_ch,out_ch,k,stride=s,padding=p,dilation=d))
        self.chomp2 = Chomp1d(p); self.relu2=nn.ReLU(); self.drop2=nn.Dropout(drop)
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
            layers.append(TemporalBlock(in_ch,ch,k,1,d,p,drop))
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

# 회귀 모델은 학습 스크립트에서 동일하게 정의되어 있다고 가정
from train_regression_models import LSTMRegressor, TCNRegressor, predict_regression

# ─── 3) Binance 서명 & 잔고 조회 ───────────────────────────
def _signed_request(http_method, url_path, payload):
    base = 'https://fapi.binance.com'
    ts = int(time.time()*1000)
    payload['timestamp'] = ts
    query = '&'.join([f"{k}={payload[k]}" for k in sorted(payload)])
    sig = hmac.new(API_SECRET, query.encode(), hashlib.sha256).hexdigest()
    headers = {'X-MBX-APIKEY': API_KEY}
    url = f"{base}{url_path}?{query}&signature={sig}"
    if http_method=='GET':
        return requests.get(url, headers=headers)
    else:
        return requests.post(url, headers=headers)

def get_usdc_balance():
    r = _signed_request('GET', '/fapi/v2/balance', {})
    r.raise_for_status()
    for e in r.json():
        if e['asset']=='USDC':
            return float(e['balance'])
    return 0.0

# ─── 4) 실시간 1m 봉 데이터 fetch ─────────────────────────
def get_latest_data(symbol, limit=61):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol":symbol,"interval":"1m","limit":limit}
    df = pd.DataFrame(requests.get(url,params=params).json(),
                      columns=["t","o","h","l","c","v","x","y","z","a","b","ignore"])
    df = df[["o","h","l","c"]].astype(float)
    df.columns=["Open","High","Low","Close"]
    df.ta.ema(length=10,append=True)
    df.ta.rsi(length=14,append=True)
    df.dropna(inplace=True)
    return df

# ─── 5) 앙상블 예측 함수 ─────────────────────────────────
def predict_ensemble(tcn,lstm,scaler,df,th_t,th_l,device):
    seq = df[['High','Low','Open','Close','EMA_10','RSI_14']].values[-60:]
    xb = torch.tensor(scaler.transform(seq),dtype=torch.float32).unsqueeze(0).to(device)
    tcn.eval(); lstm.eval()
    with torch.no_grad(), autocast():
        pt = torch.sigmoid(tcn(xb).squeeze()).item()
        pl = torch.sigmoid(lstm(xb).squeeze()).item()
    pr = (pt+pl)/2
    return 1 if pt>=th_t and pl>=th_l else 0, pr, pt, pl

# ─── 6) 시장가 주문 함수 ─────────────────────────────────
def place_order(symbol, side, qty):
    """
    side: 'BUY' or 'SELL'
    qty: 계약 수량 (USDC 마진이면 qty=USDC 금액)
    """
    r = _signed_request('POST','/fapi/v1/order',
                        {'symbol':symbol,
                         'side':side,
                         'type':'MARKET',
                         'quantity':qty})
    r.raise_for_status()
    return r.json()

# ─── 7) 메인 실행부 ─────────────────────────────────────
if __name__=='__main__':
    symbol    = "1000PEPEUSDC"
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 & 스케일러 로드
    tcn = TCNBinaryClassifier(6).to(device)
    lst = BinaryLSTM(6).to(device)
    tcn.load_state_dict(torch.load(f"{symbol}_TCN_best.pth",map_location=device))
    lst.load_state_dict(torch.load(f"{symbol}_LSTM_best.pth",map_location=device))
    scaler = joblib.load(f"{symbol}_binary_scaler.pkl")
    th_t   = float(np.load(f"{symbol}_TCN_thresh.npy"))
    th_l   = float(np.load(f"{symbol}_LSTM_thresh.npy"))

    # 회귀 모델
    reg_s = joblib.load(f"{symbol}_reg_scaler.pkl")
    reg_m = LSTMRegressor(6).to(device)
    reg_m.load_state_dict(torch.load(f"{symbol}_LSTM_reg_best.pth",map_location=device))
    reg_m.eval()

    # 손익 임계치 (레버리지 20: 예: ±0.25%), 손절도 설정
    TP = 0.25  # +0.25% 이익 시 청산
    SL = 0.15  # -0.15% 손실 시 청산

    print(f"▶️ Real-time Trading Bot Start: {symbol}")
    while True:
        try:
            # 1) 현재 USDC 잔고 확인
            bal = get_usdc_balance()
            if bal<=0:
                print("잔고 부족, 60초 대기…")
                time.sleep(60); continue

            # 2) 1m 데이터, 직전 종가
            df = get_latest_data(symbol)
            prev = df['Close'].iloc[-2]

            # 3) Classifier 앙상블
            df_in = df.iloc[:-1]
            pred_dir, avg_p, pt, pl = predict_ensemble(tcn,lst,scaler,df_in,th_t,th_l,device)
            side = 'BUY' if pred_dir==1 else 'SELL'

            # 4) Regressor 예측 → 기대변화%
            dpct = predict_regression(reg_m, reg_s, df, device)
            # 기대 수익률(%)에 수익·손절조건도 같이 고려
            # 5) 진입 판단
            enter = ((pred_dir==1 and dpct>=TP) or
                     (pred_dir==0 and dpct<=-TP))

            print(f"[{datetime.now():%H:%M}] Dir={side} ClsProb={avg_p:.1%} RegΔ={dpct:+.2f}% Enter={enter}")
            if enter:
                # 6) 시장가 진입
                order = place_order(symbol, side, qty=bal)
                print("  -> 진입 주문:", order['side'], order['origQty'])
                entry_price = float(order['avgPrice'] if 'avgPrice' in order else order['fills'][0]['price'])
                start = time.time()

                # 7) 모니터링: TP/SL 또는 1분 타임아웃
                while True:
                    now_price = get_latest_data(symbol,limit=2)['Close'].iloc[-1]
                    change = (now_price - entry_price)/entry_price*100
                    if change>=TP or change<=-SL or time.time()-start>=60:
                        # 청산
                        close_side = 'SELL' if side=='BUY' else 'BUY'
                        ct = place_order(symbol, close_side, qty=bal)
                        print(f"  -> 청산 주문({change:+.2f}%):", ct['side'], ct['origQty'])
                        break
                    time.sleep(1)

            # 8) 1분 후 다음 사이클
            time.sleep(60)

        except Exception as e:
            print("❌ 에러:", e)
            time.sleep(60)
