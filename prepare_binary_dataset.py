#!/usr/bin/env python3
"""
prepare_binary_dataset.py

Long/Short 이진 분류용 시계열 데이터셋을 생성하고 저장합니다.
Usage: python prepare_binary_dataset.py
Outputs:
  - {symbol}_binary_scaler.pkl
  - {symbol}_binary_dataset.npz
"""
from collections import Counter
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# --- 1) K라인 Fetch 헬퍼 함수 ---
def _fetch_klines(url, symbol, interval, start_ts, end_ts):
    params = dict(symbol=symbol, interval=interval, limit=1000,
                  startTime=start_ts, endTime=end_ts)
    klines = []
    while params['startTime'] < end_ts:
        resp = requests.get(url, params=params, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        klines.extend(data)
        params['startTime'] = data[-1][0] + 1
    return klines

# --- 2) Futures→Spot API 폴백 ---
def fetch_historical_klines(symbol, interval, start_date, end_date):
    fut_url  = 'https://fapi.binance.com/fapi/v1/klines'
    spot_url = 'https://api.binance.com/api/v3/klines'

    start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
    end_ts   = int(datetime.fromisoformat(end_date).timestamp() * 1000)

    try:
        raw = _fetch_klines(fut_url, symbol, interval, start_ts, end_ts)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 451:
            print('⚠️ Futures API 451, falling back to Spot API')
            sym = symbol[:-4] + 'USDT' if symbol.endswith('USDC') else symbol
            raw = _fetch_klines(spot_url, sym, interval, start_ts, end_ts)
        else:
            raise
    df = pd.DataFrame(raw, columns=[
        'Open_time','Open','High','Low','Close','Volume',
        'Close_time','Quote_asset_volume','Num_trades',
        'Taker_buy_base_asset_vol','Taker_buy_quote_asset_vol','Ignore'
    ])
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df = df[['Open_time','Open','High','Low','Close','Volume']]
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)
    return df

# --- 3) 메인 처리 ---
def main():
    symbol     = '1000PEPEUSDC'
    interval   = '1m'
    days_back  = 365
    val_days   = 7
    seq_len    = 60

    end_date   = datetime.utcnow().date().isoformat()
    start_date = (datetime.utcnow().date() - timedelta(days=days_back)).isoformat()
    train_end  = (datetime.utcnow().date() - timedelta(days=val_days)).isoformat()
    val_start  = train_end

    print(f'▶️ Fetching klines {symbol} {interval} from {start_date} to {end_date}')
    df = fetch_historical_klines(symbol, interval, start_date, end_date)
    df.ta.ema(length=10, append=True)
    df.ta.rsi(length=14, append=True)
    df.dropna(inplace=True)

    # Binary label: 다음 종가 > 현재 종가 → 1(Long), else 0(Short)
    df['Future_Close'] = df['Close'].shift(-1)
    df = df.iloc[:-1]
    df['Label'] = (df['Future_Close'] > df['Close']).astype(int)

    features = ['High','Low','Open','Close','EMA_10','RSI_14']
    scaler   = MinMaxScaler().fit(df[features])
    data_vals= scaler.transform(df[features])
    joblib.dump(scaler, f'{symbol}_binary_scaler.pkl')
    print(f'✅ Scaler saved: {symbol}_binary_scaler.pkl')

    # Split by date
    mask_train = df['Open_time'] < train_end
    X_all = []
    y_all = []
    for i in range(len(data_vals) - seq_len):
        X_all.append(data_vals[i:i+seq_len])
        y_all.append(df['Label'].iloc[i+seq_len])
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    X_train = X_all[mask_train.values[:-seq_len]]
    y_train = y_all[mask_train.values[:-seq_len]]
    X_val   = X_all[~mask_train.values[:-seq_len]]
    y_val   = y_all[~mask_train.values[:-seq_len]]
    print(f'Train shape: X={X_train.shape}, y={y_train.shape}')
    print(f'  Val shape: X={X_val.shape}, y={y_val.shape}')

    # Save datasets
    npz_name = f'{symbol}_binary_dataset.npz'
    np.savez(npz_name,
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val)
    print(f'✅ Dataset saved: {npz_name}')

    print('\n🔍 [Train Label Distribution]')
    if len(y_train) == 0:
        print("⚠️ y_train is empty!")
    else:
        print(Counter(y_train))

    print('\n🔍 [Validation Label Distribution]')
    if len(y_val) == 0:
        print("⚠️ y_val is empty!")
    else:
        print(Counter(y_val))

    if len(y_train) == 0 or len(y_val) == 0:
        print("🚫 One of the datasets is empty. Check your date ranges and symbol data.")

if __name__ == '__main__':
    main()
