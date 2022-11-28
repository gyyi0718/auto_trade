import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler


def read_csv_data(path):
    train_path = path
    train = pd.read_csv(train_path)
    df = pd.DataFrame(data = {'high':train['high'],'low':train['low'],'open':train['open']}, columns = ['high','low','open'])

    return df

def split_data(df, num):
    new_df = df.reset_index()
    y_cnt =len(new_df['index'] %num !=0)//num
    x_ = new_df[new_df['index'] %num !=0]
    y_ = new_df[new_df['index'] !=0][new_df['index'] % num==0]
    x_arr = pd.DataFrame(data={'high':x_['high'],'low':x_['low'],'open':x_['open']}, columns=['high','low','open'])
    y_arr = pd.DataFrame(data={'high':y_['high'],'low':y_['low'],'open':y_['open']}, columns=['high','low','open'])

    return x_arr.T,y_arr.T

def test_fun():
    train_path = 'bitcoin_minute_test.csv'
    df = read_csv_data(train_path)
    x_df, y_df = split_data(df,6)

    normal_x_df = normalization(x_df)
    normal_y_df = normalization(y_df)

    x_ = np.array(normal_x_df)
    test_x = x_.reshape(-1,6)
    test_y = np.array(normal_y_df)
    print()

def normalization(data):
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(data)
    return training_data

if __name__ == '__main__':
    test_fun()

