import requests
import os
import pandas as pdcd D:\ygy_work\coin

from datetime import datetime, timedelta
today = datetime.today().strftime('%y%m%d')

def read_train_csv_data(file_path):
    train = pd.read_csv(file_path)
    df = pd.DataFrame(data = {'High':train['High'],'Low':train['Low'],'Open':train['Open'],'Close':train['Close'], 'Volume':train['Volume']},
                      columns = ['High','Low','Open','Close','Volume'])
    return df

def up_down_check(data, per_ratio, standard_range, predict_range):
    print()
    result = []
    up  = 0
    down = 0
    for i in range(len(data)-(standard_range+predict_range)):
        s_data = (data['Open'][i:i+standard_range] + data['Close'][i:i+standard_range])/2.0
        p_data = ((data['Open'][i+standard_range+1:i+standard_range+1+predict_range])
                  +(data['Close'][i + standard_range + 1:i + standard_range + 1 + predict_range]))  /2.0

        if s_data.max()+(s_data.max()*per_ratio) < p_data.max():
            result.append(1)
            #print("up")
            up += 1
        elif s_data.min()-(s_data.min()*per_ratio) > p_data.min():
            result.append(-1)
            #print("down")
            down += 1
        else:
            result.append(0)
    print("up = {}, down {}".format(up, down))

    return result

if __name__ == "__main__":
    print("check data")
    file_name = "ONTUSDT_price_minute_240425"
    file_path = "D:/ygy_work/coin/{}.csv".format(file_name)
    df = read_train_csv_data(file_path)
    up_down_check(df, 0.03, 60, 10)