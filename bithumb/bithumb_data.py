# bithumb_data_collector.py
# -*- coding: utf-8 -*-
"""
Bithumb 캔들스틱 데이터 수집기
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# 설정
# =====================
SYMBOLS = [
    "BTC", "ETH", "XRP", "DOGE", "SOL", "ADA"
]
SYMBOLS = [
    "PUMP"
]
# 간격 설정 (1m, 3m, 5m, 10m, 30m, 1h, 6h, 12h, 24h)
INTERVAL = "1m"

# 데이터 수집 기간 (최근 1년)
today = datetime.utcnow()
start_date = today - timedelta(days=365)
START = start_date.strftime("%Y-%m-%d %H:%M:%S")
END = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

SAVE_CSV = "bithumb_data_pump.csv"
SAVE_PARQUET = "bithumb_data_pump.parquet"

BASE_URL = "https://api.bithumb.com/public/candlestick"
MAX_WORKERS = min(8, len(SYMBOLS))

# 간격별 밀리초 변환
INTERVAL_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "10m": 10 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "24h": 24 * 60 * 60 * 1000,
}


def to_ms(dt):
    """datetime을 밀리초 타임스탬프로 변환"""
    return int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)


def fetch_bithumb_candlestick(session, symbol, interval="1h"):
    """
    빗썸 API에서 캔들스틱 데이터 가져오기

    API 응답 형식:
    {
        "status": "0000",
        "data": [
            [timestamp, open, close, high, low, volume],
            ...
        ]
    }
    """
    try:
        url = f"{BASE_URL}/{symbol}_KRW/{interval}"
        resp = session.get(url, timeout=10)

        if resp.status_code != 200:
            print(f"[WARN] {symbol} HTTP {resp.status_code}")
            return pd.DataFrame()

        data = resp.json()

        if data.get("status") != "0000":
            print(f"[WARN] {symbol} API 오류: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        candles = data.get("data", [])
        if not candles:
            print(f"[WARN] {symbol} 데이터 없음")
            return pd.DataFrame()

        # DataFrame 생성
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "close", "high", "low", "volume"]
        )

        # 데이터 타입 변환
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 심볼 컬럼 추가
        df.insert(1, "symbol", symbol)

        # 정렬 및 중복 제거
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        print(f"[OK] {symbol} rows={len(df)} {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        return df

    except requests.exceptions.Timeout:
        print(f"[ERR] {symbol} 타임아웃")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"[ERR] {symbol} 요청 오류: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERR] {symbol} 예외 발생: {e}")
        return pd.DataFrame()


def get_available_symbols(session):
    """거래 가능한 심볼 목록 가져오기"""
    try:
        url = "https://api.bithumb.com/public/ticker/ALL_KRW"
        resp = session.get(url, timeout=10)
        data = resp.json()

        if data.get("status") != "0000":
            print("[WARN] 심볼 목록 가져오기 실패")
            return []

        symbols = [k for k in data.get("data", {}).keys() if k not in ["date", "timestamp"]]
        print(f"[INFO] 거래 가능한 심볼 {len(symbols)}개 발견")
        return symbols

    except Exception as e:
        print(f"[WARN] 심볼 목록 로드 실패: {e}")
        return []


def main():
    print(f"[시작] 빗썸 데이터 수집")
    print(f"심볼: {', '.join(SYMBOLS)}")
    print(f"간격: {INTERVAL}")
    print(f"기간: {START} ~ {END}")
    print("-" * 60)

    start_time = time.time()
    all_df = []

    with requests.Session() as s:
        s.headers.update({
            "User-Agent": "bithumb-data-collector/1.0",
            "Accept": "application/json"
        })

        # 사용 가능한 심볼 확인 (선택사항)
        # available_symbols = get_available_symbols(s)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(fetch_bithumb_candlestick, s, sym, INTERVAL): sym
                for sym in SYMBOLS
            }

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_df.append(df)
                    time.sleep(0.1)  # API 호출 간격
                except Exception as e:
                    print(f"[ERR] {sym} 처리 중 오류: {e}")

    if not all_df:
        print("[ERROR] 수집된 데이터가 없습니다.")
        return

    # 데이터 병합 및 저장
    df_all = pd.concat(all_df, ignore_index=True)
    df_all = df_all.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # CSV 저장
    df_all.to_csv(SAVE_CSV, index=False, encoding="utf-8-sig")
    print(f"[저장] CSV: {SAVE_CSV} ({len(df_all):,} rows)")

    # Parquet 저장
    df_all.to_parquet(SAVE_PARQUET, index=False)
    print(f"[저장] Parquet: {SAVE_PARQUET} ({len(df_all):,} rows)")

    # 요약 정보
    print("\n" + "=" * 60)
    print("수집 완료 요약:")
    print(f"총 데이터: {len(df_all):,} rows")
    print(f"심볼별 데이터:")
    for sym in SYMBOLS:
        sym_df = df_all[df_all["symbol"] == sym]
        if not sym_df.empty:
            print(f"  {sym}: {len(sym_df):,} rows")

    elapsed = time.time() - start_time
    print(f"\n소요 시간: {elapsed:.2f}초")
    print("=" * 60)


if __name__ == "__main__":
    main()