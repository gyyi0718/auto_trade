# bybit_futures_daily_fast.py
# -*- coding: utf-8 -*-
"""
Bybit USDT-Perp Daily(일봉) collector — fast paging
"""

import time
import requests, os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

# =====================
# 설정
# =====================
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "SHIBUSDT", "LINKUSDT", "BCHUSDT"
]
SYMBOLS = [
    "HUSDT"#, "ETHUSDT", "BNBUSDT", "SOLUSDT",  "DOGEUSDT", "ADAUSDT", "XRPUSDT"
   ]

INTERVAL = "D"  # 일봉 (Daily)
CATEGORY = "linear"  # Bybit Futures USDT Perpetual

# 최근 3년치 데이터 (일봉은 더 긴 기간도 가능)
today = datetime.utcnow()
start_date = today - timedelta(days=1825)  # 5년
START = start_date.strftime("%Y-%m-%d %H:%M:%S")
END = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

SAVE_CSV = "bybit_futures_daily_h.csv"
SAVE_PARQUET = "bybit_futures_daily_h.parquet"

BASE_URL = "https://api.bybit.com/v5/market/kline"
PAGE_LIMIT = 1000
MAX_WORKERS = min(8, len(SYMBOLS))

TESTNET = str(os.getenv("BYBIT_TESTNET", "0")).lower() in ("1", "true", "y", "yes")
mkt = HTTP(testnet=TESTNET, timeout=10, recv_window=5000)

_TRADABLE_LINEAR = None


def get_tradable_linear_symbols():
    global _TRADABLE_LINEAR
    if _TRADABLE_LINEAR is not None:
        return _TRADABLE_LINEAR
    try:
        info = mkt.get_instruments_info(category="linear")
        lst = info.get("result", {}).get("list", []) or []
        syms = {
            it["symbol"].upper().strip()
            for it in lst
            if str(it.get("status", "Trading")).lower() == "trading"
               and "USDT" in it.get("settleCoin", "USDT")
               and str(it.get("contractType", "LinearPerpetual")).lower().startswith("linear")
        }
        _TRADABLE_LINEAR = syms
        print(f"[INFO] tradable linear symbols loaded: {len(syms)}")
        return syms
    except Exception as e:
        print(f"[WARN] instruments-info load failed: {e}")
        _TRADABLE_LINEAR = set()
        return _TRADABLE_LINEAR


def _symbol_ok(sym):
    syms = get_tradable_linear_symbols()
    if syms:
        return sym.upper().strip() in syms
    s = sym.upper().strip()
    return s.endswith("USDT") and len(s) >= 6


def to_ms(dt):
    return int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)


def fetch_ohlcv_descending(session, symbol, start_ms, end_ms, interval="D"):
    """
    내림차순 응답 특성에 맞춰 end 커서를 과거로 이동
    일봉: 각 캔들은 86400초(24시간) 간격
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

        pages.extend(rows)

        # 가장 과거 timestamp
        oldest_ts = min(int(x[0]) for x in rows)

        # 1일 = 86400초 = 86,400,000ms
        end_cursor = oldest_ts - 86_400_000

        if len(rows) < PAGE_LIMIT and end_cursor < start_ms:
            break

    if not pages:
        return pd.DataFrame()

    df = pd.DataFrame(pages, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= pd.to_datetime(start_ms, unit="ms", utc=True)) &
            (df["timestamp"] < pd.to_datetime(end_ms, unit="ms", utc=True))].reset_index(drop=True)
    df.insert(1, "symbol", symbol)
    return df


def main():
    start_ms = to_ms(START)
    end_ms = to_ms(END)

    all_df = []
    with requests.Session() as s:
        s.headers.update({"User-Agent": "bybit-daily-collector/1.0"})
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

    # Daily 데이터는 date 컬럼으로 변환 (시간 제거)
    df_all['date'] = df_all['timestamp'].dt.date
    df_all = df_all.drop(columns=['timestamp'])
    df_all = df_all[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

    df_all.to_csv(SAVE_CSV, index=False)
    df_all.to_parquet(SAVE_PARQUET, index=False)
    print(f"[DONE] Saved {len(df_all)} rows to {SAVE_CSV} / {SAVE_PARQUET}")


if __name__ == "__main__":
    main()