import ccxt
import time
from datetime import datetime, timedelta
import logging
from binance

def get_all_symbols(exchange):
    markets = exchange.load_markets()
    return [symbol for symbol in markets]


def get_top_volume_increase(exchange, symbols):
    volumes = {}
    now = int(datetime.timestamp(datetime.now()) * 1000)
    yesterday = int(datetime.timestamp(datetime.now() - timedelta(days=1)) * 1000)

    for symbol in symbols:
        try:
            ticker_now = exchange.fetch_ohlcv(symbol, '1d', limit=1, since=now)[0][5]
            ticker_yesterday = exchange.fetch_ohlcv(symbol, '1d', limit=1, since=yesterday)[0][5]
            volume_increase = ticker_now - ticker_yesterday
            volumes[symbol] = volume_increase
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            volumes[symbol] = None

    return dict(sorted(volumes.items(), key=lambda x: x[1], reverse=True))


def main():
    exchange = ccxt.binance()
    symbols = get_all_symbols(exchange)

    # API 속도 제한을 고려한 요청 간격 계산
    symbols_per_request = 100
    sleep_time = (len(symbols) / symbols_per_request) * 6

    while True:
        top_volume_increase = get_top_volume_increase(exchange, symbols)

        for symbol, volume_increase in top_volume_increase.items():
            print(f"{symbol}: {volume_increase}")

        time.sleep(sleep_time)  # API 속도 제한을 고려한 요청 간격


if __name__ == "__main__":
    main()