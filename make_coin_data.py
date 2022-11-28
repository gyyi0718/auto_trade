import pandas as pd
import cryptocompare
from datetime import date, timedelta
import datetime
import time
from typing import Union, Optional, List, Dict

today = datetime.datetime.today().strftime('%y%m%d')

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def bitcoin_date():
    cryptocompare.get_historical_price_day('BTC', 'USD', limit=24, exchange='CCCAGG',
                                           toTs=datetime.datetime(2019, 6, 6))
    # or
    cryptocompare.get_historical_price_day('BTC', 'USD', limit=24, exchange='CCCAGG',
                                           toTs=datetime.datetime(1514732400))

    cryptocompare.get_historical_price_hour('BTC', 'USD', limit=24, exchange='CCCAGG',
                                            toTs=datetime.datetime(2019, 6, 6, 12))

    cryptocompare.get_historical_price_minute('BTC', 'USD', limit=24, exchange='CCCAGG', toTs=datetime.datetime.now())


def main():
    start_date = date(2018, 1, 1)
    end_date = date.today()-timedelta(days=7)
    save_path = 'bitcoin.csv'
    for single_date in daterange(start_date, end_date):
        hour_obj = cryptocompare.get_historical_price_hour('BTC', 'USD', limit=24, exchange='CCCAGG',toTs=single_date)
        for i in hour_obj:
            df = pd.DataFrame(data=i,columns=i.keys(), index=[0])
            df.to_csv(save_path,mode='a',sep=',',index=False,header=False)


Timestamp = Union[datetime.datetime, datetime.date, int, float]
def _format_timestamp(timestamp: Timestamp) -> int:
    """
    Format the timestamp depending on its type and return
    the integer representation accepted by the API.

    :param timestamp: timestamp to format
    """
    if isinstance(timestamp, datetime.datetime) or isinstance(timestamp, datetime.date):
        return int(time.mktime(timestamp.timetuple()))
    return int(timestamp)


def write_minutes_data():
    start_date = date(2018, 1, 1)
    end_date = date(2022, 11, 27)
    start_minutes = _format_timestamp(start_date)
    end_minutes = _format_timestamp(end_date)

    save_path = 'bitcoin_minute_%s.csv'%today
    for minute_time in range(start_minutes, end_minutes,60):
        time_obj = cryptocompare.get_historical_price_minute('BTC', 'USD', limit=1440, exchange='CCCAGG', toTs=minute_time)
        for i in time_obj:
            df = pd.DataFrame(data=i, columns=i.keys(), index=[0])
            df.to_csv(save_path, mode='a', sep=',', index=False, header=False)

def write_minutes_data2():
    start_date = date(2022, 11, 27)
    start_minutes = _format_timestamp(start_date)
    end_minutes = _format_timestamp(start_date+3600*12)
    save_path = 'minute_pred_%s.csv'%today
    for minute_time in range(start_minutes, end_minutes,60):
        time_obj = cryptocompare.get_historical_price_minute('BTC', 'KRW', limit=1440, exchange='CCCAGG', toTs=minute_time)
        for i in time_obj:
            df = pd.DataFrame(data=i, columns=i.keys(), index=[0])
            if minute_time==start_minutes:
                df.to_csv(save_path, mode='a', sep=',', index=False, header=True)
            else:
                df.to_csv(save_path, mode='a', sep=',', index=False, header=False)

def write_minutes_data3():
    start_date = date(2018, 1, 1)
    end_date = date(2022,11,27)
    start_minutes = _format_timestamp(start_date)
    end_minutes = _format_timestamp(end_date)
    save_path = 'hour_%s.csv'%today
    for minute_time in range(start_minutes, end_minutes,3600):
        time_obj = cryptocompare.get_historical_price_hour('BTC', 'KRW', limit=1440, exchange='CCCAGG', toTs=minute_time)
        for i in time_obj:
            df = pd.DataFrame(data=i, columns=i.keys(), index=[0])
            if minute_time==start_minutes:
                df.to_csv(save_path, mode='a', sep=',', index=False, header=True)
            else:
                df.to_csv(save_path, mode='a', sep=',', index=False, header=False)



if __name__ == '__main__':
    write_minutes_data3()

