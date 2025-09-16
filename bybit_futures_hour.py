# bybit_futures_klines_fast.py
# -*- coding: utf-8 -*-
"""
Bybit USDT-Perp collector — fast paging
- 내림차순 응답 특성에 맞춰 end 커서를 과거로 이동(역방향 페이징)
- requests.Session 재사용 + 멀티스레딩
- 중복 제거 & 벡터화 변환
- ✅ INTERVAL 일반화 (1m~1h~일봉 등), 커서(step) 자동 계산
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
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]
# 🔽 여기만 바꾸면 1시간봉 수집됨: "60"
# (예: 1m="1", 3m="3", 5m="5", 15m="15", 30m="30", 60m="60", 120m="120", 240m="240", 일봉="D", 주봉="W", 월봉="M")
INTERVAL = "60"          # 1시간봉
CATEGORY = "linear"      # Bybit Futures USDT Perpetual

today = datetime.utcnow()
start_date = today - timedelta(days=1000)   # 6개월치 예시 (원하면 조정)
START = start_date.strftime("%Y-%m-%d %H:%M:%S")
END   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

BASE_URL = "https://api.bybit.com/v5/market/kline"
PAGE_LIMIT = 1000
MAX_WORKERS = min(8, len(SYMBOLS))  # 적당한 동시성

def interval_to_ms(interval: str) -> int:
    """Bybit v5 interval 문자열을 ms로 변환."""
    if interval in {"D", "d"}:
        return 24 * 60 * 60 * 1000
    if interval in {"W", "w"}:
        return 7 * 24 * 60 * 60 * 1000
    if interval in {"M"}:  # 월
        # 월 단위는 길이가 가변이라 페이징 스텝만 대략치(30일)로 사용
        return 30 * 24 * 60 * 60 * 1000
    # 분 또는 시간(분 단위 숫자) 처리
    try:
        mins = int(interval)
        return mins * 60 * 1000
    except ValueError:
        raise ValueError(f"Unsupported interval: {interval}")

def to_ms(dt: str) -> int:
    # 명시적 UTC로 해석
    return int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)

def fetch_ohlcv_descending(session: requests.Session, symbol: str, start_ms: int, end_ms: int, interval="1") -> pd.DataFrame:
    """
    Bybit 응답은 최신→과거 내림차순.
    => start는 고정, end 커서를 '가장 오래된 ts - PERIOD_MS'로 당겨가며 페이징.
    """
    period_ms = interval_to_ms(interval)
    end_cursor = end_ms
    pages = []

    while end_cursor >= start_ms:
        params = {
            "category": CATEGORY,
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_cursor,
            "limit": PAGE_LIMIT
        }
        resp = session.get(BASE_URL, params=params, timeout=15)
        j = resp.json()
        if j.get("retCode", 0) != 0:
            print(f"[WARN] {symbol} retCode={j.get('retCode')} retMsg={j.get('retMsg')}")
            break

        rows = j.get("result", {}).get("list", [])
        if not rows:
            break

        # rows: [[ts, open, high, low, close, volume, turnover], ...] (문자열)
        pages.extend(rows)

        # 이번 페이지의 가장 과거 ts 찾기
        oldest_ts = min(int(x[0]) for x in rows)

        # 다음 페이지는 이번 가장 과거 직전 캔들까지로 end를 당김
        end_cursor = oldest_ts - period_ms

        # 남은 구간이 limit 미만이라면 다음 한 번으로 끝날 수 있음
        if len(rows) < PAGE_LIMIT and end_cursor < start_ms:
            break

    if not pages:
        return pd.DataFrame()

    # DataFrame 생성 (벡터화 변환)
    df = pd.DataFrame(pages, columns=["timestamp","open","high","low","close","volume","turnover"])

    # 타입 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ["open","high","low","close","volume","turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 중복 제거, 시간 오름차순 정렬, 범위 필터
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= pd.to_datetime(start_ms, unit="ms", utc=True)) &
            (df["timestamp"] <  pd.to_datetime(end_ms,   unit="ms", utc=True))].reset_index(drop=True)
    df.insert(1, "symbol", symbol)
    return df

def main():
    start_ms = to_ms(START)
    end_ms   = to_ms(END)

    suffix = INTERVAL.lower()
    save_csv = f"bybit_futures_{suffix}.csv"
    save_parquet = f"bybit_futures_{suffix}.parquet"

    all_df = []
    with requests.Session() as s:
        s.headers.update({"User-Agent": "bybit-fast-collector/1.1"})
        # 심볼 병렬 수집
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_ohlcv_descending, s, sym, start_ms, end_ms, INTERVAL): sym for sym in SYMBOLS}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    df = fut.result()
                    if df.empty:
                        print(f"[WARN] {sym} no data.")
                    else:
                        print(f"[OK] {sym} rows={len(df)} {df['timestamp'].min()} ~ {df['timestamp'].max()}")
                        all_df.append(df)
                except Exception as e:
                    print(f"[ERR] {sym} {e}")

    if not all_df:
        print("[ERROR] No data collected.")
        return

    df_all = pd.concat(all_df, ignore_index=True)
    df_all.to_csv(save_csv, index=False)
    df_all.to_parquet(save_parquet, index=False)
    print(f"[DONE] Saved {len(df_all)} rows to {save_csv} / {save_parquet}")

if __name__ == "__main__":
    main()
