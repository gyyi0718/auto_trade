import asyncio
import csv
import json
import time
import websockets
import pandas as pd
import ssl
import cryptocompare
import datetime

key = '886bd978c221610f1e9836ae9d0d153c6786b832307143f58d69a566a47fdda0'
cryptocompare.cryptocompare._set_api_key_parameter(key)


def main(path):
    df = pd.read_csv(path)
    df = df.reset_index()
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df[['open', 'high', 'low', 'close', 'Volume USDT']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[::-1]
    df = df.astype('float')


def cryptodatadownload_data(ticker,save_path):
    '''
    cryptodatadownload.com의 시간봉 데이터를 가져옴
    제공 데이터 -> 비트/이더/리플/링크/라이트/에이다/이오스/빗캐/유니/스시/폴카닷/븐브/테조스/트론
    columns = ['Open','High','Low','Close','Volume']
    '''

    ssl._create_default_https_context = ssl._create_unverified_context
    url = 'https://www.cryptodatadownload.com/cdd/FTX_Futures_' + ticker.split('-')[0] + 'PERP_1h.csv'
    df = pd.read_csv(url)
    df = df.reset_index()
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df[['open', 'high', 'low', 'close', 'Volume USDT']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[::-1]
    df = df.astype('float')
    df.to_csv(save_path)

    return df


PARAMS = {
  "action": "subscribe",
  "params": {
	"symbols": "ETH/BTC"
  }
}
PARAMS2 ={
    "action": "SubAdd",
    "subs": ["5~CCCAGG~BTC~USD", "0~Coinbase~ETH~USD", "2~Binance~BTC~USDT"]
}

async def run():
    url_ = 'wss://streamer.cryptocompare.com/v2/quotes/price?apikey=886bd978c221610f1e9836ae9d0d153c6786b832307143f58d69a566a47fdda0'

    url__ = 'wss://ws.twelvedata.com/v1/quotes/price?apikey=8fc09b27ef394f448ad5c18d33fb6635'
    websocket = await websockets.connect(url_, ping_interval=10)
    await websocket.send(json.dumps(PARAMS2))

    with open('eth_btc.csv', 'a') as file:
        writer = csv.writer(file)
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(time.strftime('%c', time.localtime(time.time())))
            print(data.keys())
            print(data.values())

            writer.writerow(data.keys())
            writer.writerow(data.values())


def crypto_read():
    a = cryptocompare.get_price('BTC', currency='KRW')
    url_ = 'https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=KRWapikey=886bd978c221610f1e9836ae9d0d153c6786b832307143f58d69a566a47fdda0'
    #websocket = await websockets.connect(url_, ping_interval=10)

    # 저장소 ex['Korbit'], ex['Bithumb']
    ex = cryptocompare.get_exchanges()
    b = cryptocompare.get_coin_list(format=False)

    c = cryptocompare.get_price('BTC')
    # or
    d = cryptocompare.get_price('BTC', currency='KRW', full=True)
    # or
    e = cryptocompare.get_price(['BTC', 'ETH'], ['EUR', 'GBP'])

    f = cryptocompare.get_historical_price('BTC', timestamp=datetime.datetime(2017, 6, 6), exchange='CCCAGG')

    day = cryptocompare.get_historical_price_day('BTC', currency='KRW')
    day1 = cryptocompare.get_historical_price_day('BTC', currency='KRW', limit=30)
    day2 = cryptocompare.get_historical_price_day('BTC', 'KRW', limit=24, exchange='CCCAGG',toTs=datetime.datetime(2019, 6, 6))

    # or

    hour = cryptocompare.get_historical_price_hour('BTC', currency='KRW')
    hour1 = cryptocompare.get_historical_price_hour('BTC', currency='KRW', limit=24)
    hour2 = cryptocompare.get_historical_price_hour('BTC', 'KRW', limit=24, exchange='CCCAGG')
    hour3 = cryptocompare.get_historical_price_hour('BTC', 'KRW', limit=24, exchange='CCCAGG',toTs=datetime.datetime(2019, 6, 6, 12))
    # or


    minute =cryptocompare.get_historical_price_minute('BTC', currency='KRW')
    minute1 = cryptocompare.get_historical_price_minute('BTC', currency='KRW', limit=1440)
    minute2 = cryptocompare.get_historical_price_minute('BTC', 'KRW', limit=24, exchange='CCCAGG', toTs=datetime.datetime.now())
    minute3 = cryptocompare.get_historical_price_minute('BTC', 'KRW', limit=24, exchange='CCCAGG', toTs=datetime.datetime(2019, 6, 6,21,40))

    average = cryptocompare.get_avg('BTC', currency='KRW', exchange='Kraken')

    print()



if __name__ == "__main__":
    stock_name =  'BTC-PERP'
    save_path = 'D:/btc_perp.csv'
    crypto_read()
    #asyncio.run(run())