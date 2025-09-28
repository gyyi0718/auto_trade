# ✅ 목적: DeepAR 모델 학습용 시계열 수치 예측 데이터 생성 (log return 기반) — Upbit 버전

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import os
import time
import sys

# === 사용자 설정 ===
SYMBOL   = "KRW-ETH"   # 업비트 마켓 코드 예: KRW-BTC, KRW-SOL
INTERVAL = 1           # 분봉 단위: 1,3,5,10,15,30,60,240
DAYS     = 120

now = datetime.now()
date_str = now.strftime("%y%m%d")

# === Upbit 분봉 데이터 수집 ===
def fetch_upbit_klines(market: str, unit_minutes: int, days: int) -> pd.DataFrame:
    """
    Upbit /v1/candles/minutes/{unit} 사용.
    최신→과거 순으로 반환되므로 to 파라미터로 페이지 다운, 최종적으로 시간 오름차순 정렬.
    """
    base_url = f"https://api.upbit.com/v1/candles/minutes/{unit_minutes}"
    end_dt_utc   = datetime.utcnow().replace(tzinfo=timezone.utc)
    start_dt_utc = end_dt_utc - timedelta(days=days)
    to_cursor = end_dt_utc

    all_rows = []
    max_attempts = 20000  # 넉넉히
    while to_cursor > start_dt_utc and max_attempts > 0:
        params = {
            "market": market,
            "count": 200,  # 최대 200개
            "to": to_cursor.strftime("%Y-%m-%dT%H:%M:%S") + "Z",  # UTC ISO8601
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            data = res.json()

            if isinstance(data, dict) and data.get("error"):
                print(f"🚫 Upbit API 오류: {data}")
                break
            if not data:
                print("⚠️ 빈 응답 발생. 종료합니다.")
                break

            all_rows.extend(data)

            # 다음 페이지 커서는 가장 오래된 캔들의 UTC 시각
            oldest_utc = data[-1]["candle_date_time_utc"]  # 'YYYY-MM-DDTHH:MM:SS'
            to_cursor = datetime.strptime(oldest_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            max_attempts -= 1
            time.sleep(0.2)  # 레이트리밋 여유

        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # timestamp(ms) 또는 candle_date_time_utc 사용 → Timestamp(UTC) 생성
    if "timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["Timestamp"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)

    df.rename(columns={
        "trade_price": "Close",
        "candle_acc_trade_volume": "Volume",
    }, inplace=True)

    # 필요한 컬럼만, 시간 오름차순 정렬
    df = df[["Timestamp", "Close", "Volume"]].astype({"Close": float, "Volume": float})
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # 기간 필터(안전)
    df = df[df["Timestamp"] >= start_dt_utc].reset_index(drop=True)
    return df

# === 1. 데이터 수집 ===
print("📡 Upbit에서 현물 분봉 데이터 수집 중...")
df = fetch_upbit_klines(SYMBOL, INTERVAL, DAYS)

if df.empty:
    print("🚫 데이터를 가져오지 못했습니다.")
    sys.exit(1)

# === 2. 로그 수익률 생성 ===
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# === 3. DeepAR 입력용 컬럼 생성 ===
df["time_idx"]  = np.arange(len(df))
df["series_id"] = SYMBOL

# === 4. 저장 ===
output_df = df[["time_idx", "series_id", "Close", "Volume", "log_return"]].copy()
output_df.columns = ["time_idx", "series_id", "close", "volume", "log_return"]

out_path = f"{SYMBOL}_deepar_input_{date_str}.csv"
output_df.to_csv(out_path, index=False)
print(f"✅ {out_path} 생성 완료 (샘플 수: {len(output_df)})")
