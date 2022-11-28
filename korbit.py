import pykorbit

arr = ['BTC', 'ETH', 'ETC', 'XRP', 'BCH', 'BTG', 'LTC', 'ZIL']


def main():
    #가상화폐 거리를 위한 티커, 티커 조회
    tickers = pykorbit.get_tickers()
    print(tickers)

    # 현재가 조회
    price = pykorbit.get_current_price("BTC")

    #과거 데이터 조회
    df = pykorbit.get_ohlc("BTC")

    #과거 5일 전 데이터
    five_days_df = pykorbit.get_ohlc("BTC", period=5)
    print(five_days_df)

    # 호가조회
    orderbook = pykorbit.get_orderbook("BTC")
    #print(orderbook)


if __name__ =="__main__":
    main()