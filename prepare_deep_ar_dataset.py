# ✅ 목적: DeepAR 모델 학습용 시계열 수치 예측 데이터 생성 (log return 기반)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import time

# === 사용자 설정 ===
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
DAYS = 120
now = datetime.now()
date_str = now.strftime("%y%m%d")
# === Binance 선물 API를 통한 1분봉 데이터 수집 ===
def fetch_binance_klines(symbol, interval, days):
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    max_attempts = 1000

    while start_time < end_time and max_attempts > 0:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            res = requests.get(base_url, params=params, timeout=10)
            data = res.json()

            if isinstance(data, dict) and 'code' in data:
                print(f"🚫 Binance API 오류: {data}")
                break

            if not data:
                print("⚠️ 빈 응답 발생. 종료합니다.")
                break

            all_data.extend(data)
            start_time = data[-1][0] + 1
            max_attempts -= 1
            time.sleep(0.3)

        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            break

    df = pd.DataFrame(all_data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "Quote_asset_volume", "Number_of_trades",
        "Taker_buy_base_asset_volume", "Taker_buy_quote_asset_volume", "Ignore"
    ])
    return df

# === 1. 데이터 수집 ===
print("📡 Binance에서 선물 데이터 수집 중...")
df = fetch_binance_klines(SYMBOL, INTERVAL, DAYS)

if df.empty:
    print("🚫 데이터를 가져오지 못했습니다.")
    exit()

# === 2. 필요한 컬럼 정제 및 추가 ===
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
df = df[['Timestamp', 'Close', 'Volume']].astype({'Close': float, 'Volume': float})
df = df.dropna().reset_index(drop=True)

# === 3. 로그 수익률 생성 ===
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

# === 4. DeepAR 입력용 컬럼 생성 ===
df['time_idx'] = np.arange(len(df))
df['series_id'] = SYMBOL

# === 5. 저장 ===
output_df = df[['time_idx', 'series_id', 'Close', 'Volume', 'log_return']]
output_df.columns = ['time_idx', 'series_id', 'close', 'volume', 'log_return']
output_df.to_csv(f"{SYMBOL}_deepar_input_{date_str}.csv", index=False)
print(f"✅ {SYMBOL}_deepar_input_test.csv 생성 완료 (샘플 수: {len(output_df)})")
