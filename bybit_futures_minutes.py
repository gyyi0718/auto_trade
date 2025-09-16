# bybit_futures_minutes_fast.py
# -*- coding: utf-8 -*-
"""
Bybit USDT-Perp 1m collector — fast paging
- 내림차순 응답 특성에 맞춰 end 커서를 과거로 이동(역방향 페이징)
- requests.Session 재사용 + 멀티스레딩
- 중복 제거 & 벡터화 변환
"""

import time
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
# =====================
# 설정
# =====================
SYMBOLS2 = [
    "ETHUSDT","BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "BNBUSDT","ADAUSDT","LINKUSDT","UNIUSDT","TRXUSDT",
    "LTCUSDT","MNTUSDT","SUIUSDT","1000PEPEUSDT",
    "XLMUSDT","ARBUSDT","APTUSDT","OPUSDT","AVAXUSDT"
]

SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT"
]
INTERVAL = "1"   # 1m
CATEGORY = "linear"  # Bybit Futures USDT Perpetual
today = datetime.utcnow()
start_date = today - timedelta(days=60)
START = start_date.strftime("%Y-%m-%d %H:%M:%S")
END   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

SAVE_CSV = "bybit_futures_1m.csv"
SAVE_PARQUET = "bybit_futures_1m.parquet"

BASE_URL = "https://api.bybit.com/v5/market/kline"
PAGE_LIMIT = 1000
MAX_WORKERS = min(8, len(SYMBOLS))  # 적당한 동시성

def to_ms(dt: str) -> int:
    # 명시적 UTC로 해석
    return int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)

def fetch_ohlcv_descending(session: requests.Session, symbol: str, start_ms: int, end_ms: int, interval="1") -> pd.DataFrame:
    """
    Bybit 응답은 최신→과거 내림차순.
    => start는 고정, end 커서를 '가장 오래된 ts - 60s'로 당겨가며 페이징.
    호출 횟수 ~ ceil(총분수/limit)
    """
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
        resp = session.get(BASE_URL, params=params, timeout=10)
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

        # 다음 페이지는 이번 가장 과거 직전 분까지로 end를 당김
        end_cursor = oldest_ts - 60_000

        # 남은 구간이 limit 미만이라면 다음 한 번으로 끝나므로 루프 빠르게 종료 가능
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

    # 중복 제거(혹시 모를 경계 중복 방지), 시간 오름차순 정렬, 범위 필터
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= pd.to_datetime(start_ms, unit="ms", utc=True)) &
            (df["timestamp"] <  pd.to_datetime(end_ms,   unit="ms", utc=True))].reset_index(drop=True)
    df.insert(1, "symbol", symbol)
    return df

def main():
    start_ms = to_ms(START)
    end_ms   = to_ms(END)

    all_df = []
    with requests.Session() as s:
        s.headers.update({"User-Agent": "bybit-fast-collector/1.0"})
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
    df_all.to_csv(SAVE_CSV, index=False)
    df_all.to_parquet(SAVE_PARQUET, index=False)
    print(f"[DONE] Saved {len(df_all)} rows to {SAVE_CSV} / {SAVE_PARQUET}")

if __name__ == "__main__":
    main()
