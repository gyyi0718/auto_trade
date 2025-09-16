import os.path
import pandas as pd
import datetime
from datetime import timedelta
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler,StandardScaler,RobustScaler
import tensorflow as tf
import math
from keras.losses import Huber
from keras.optimizers import SGD, Adam
import time
from typing import Union, Optional, List, Dict
from pickle import dump, load
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import LambdaCallback

today = datetime.datetime.today().strftime('%y%m%d')
Timestamp = Union[datetime.datetime, datetime.date, int, float]
s_price = 66278
# Set up the API endpoint URL
URL = "https://api.binance.com/api/v3/klines"

# Define the symbol and interval
symbol = "BTCUSDT"
interval = "1m"

# Calculate the start and end dates
end_date = datetime.datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=365)

# Convert start and end dates to Unix timestamps
start_ts = int(time.mktime(start_date.timetuple())) * 1000
end_ts = int(time.mktime(end_date.timetuple())) * 1000

# Define the parameters for the API request
params = {
    "symbol": symbol,
    "interval": interval,
    "startTime": start_ts,
    "endTime": end_ts,
    "limit": 10000
}

def _format_timestamp(timestamp: Timestamp) -> int:
    """
    Format the timestamp depending on its type and return
    the integer representation accepted by the API.

    :param timestamp: timestamp to format
    """
    if isinstance(timestamp, datetime.datetime) or isinstance(timestamp, datetime.date):
        return int(time.mktime(timestamp.timetuple()))
    return int(timestamp)


def read_json_data(path):
    train_path = path
    train = pd.read_json(train_path)
    train.info()
    price_list = []

    #for i in range(len(train)):
    for i in range(0,1000):
        price = train['market-price'][i]['y']
        price_list.append(price)
    df = pd.DataFrame(data = {'price':price_list}, columns = ['price'])
    temp_price = df['price'].diff(periods=1)
    temp_price[0] = 0
    new_df = pd.DataFrame(data = {'diff':temp_price}, columns = ['class','diff'])
    new_df['class'] = [1 if t else 0 for t in list(new_df['diff'] > 0)]

    return new_df

def read_train_csv_data(path):
    train_path = path
    train = pd.read_csv(train_path)
    df = pd.DataFrame(data = {'High':train['High'],'Low':train['Low'],'Open':train['Open'],'Close':train['Close']}, columns = ['High','Low','Open','Close'])
    #global s_price
    #s_price = train['Open'][0]
    return df


def split_data(df,num):
    column = list(df.keys())
    x_ = pd.DataFrame(df.iloc[0])
    y_ = pd.DataFrame(df.iloc[num])
    #for i in range(len(df)):
    for i in range(0,len(df)):
        if i % num == 0 and i !=0:
            y_ = pd.concat([y_, df.iloc[i]],ignore_index=True,axis=1)
        elif i%num !=0 and i !=0:
            x_ = pd.concat([x_, df.iloc[i]],ignore_index=True,axis=1)

    return x_.T,y_.T


def temp_split_data(df, num):
    new_df = df.copy()
    new_df = df.reset_index()
    y_cnt =len(new_df['index'] %num !=0)//num
    x_ = new_df[new_df['index'] %num !=0]
    y_ = new_df[new_df['index'] !=0][new_df['index'] % num==0]
    x_arr = pd.DataFrame(data={'High':x_['High'],'Low':x_['Low'],'Open':x_['Open']}, columns=['High','Low','Open'])
    y_arr = pd.DataFrame(data={'High':y_['High'],'Low':y_['Low'],'Open':y_['Open']}, columns=['High','Low','Open'])

    return x_arr.T,y_arr.T


def temp_split_data2(df, num):
    new_df = df.copy()
    new_df = df.reset_index()
    y_cnt =len(new_df['index'] %num !=0)//num
    #x_ = new_df[new_df['index'] %num !=0]
    x_ = new_df
    y_ = new_df[new_df['index'] !=0][new_df['index'] % num==0]
    x_arr = pd.DataFrame(data={'High':x_['High'],'Low':x_['Low'],'Open':x_['Open']}, columns=['High','Low','Open'])
    y_arr = pd.DataFrame(data={'High':y_['High'],'Low':y_['Low'],'Open':y_['Open']}, columns=['High','Low','Open'])

    return x_arr.T,y_arr.T

def five_minutes_split_data(df, num):
    new_df = df.reset_index()
    y_train = []
    x_high = np.zeros(0)
    x_low = np.zeros(0)
    x_open = np.zeros(0)
    x_close = np.zeros(0)

    for i in range(0, (len(df)-len(df)%6)-num):
        x_high = np.concatenate((x_high, np.array(new_df['High'][i:i+num-1])),axis=0)
        x_low = np.concatenate((x_low, np.array(new_df['Low'][i:i+num-1])),axis=0)
        x_open = np.concatenate((x_open, np.array(new_df['Open'][i:i+num-1])),axis=0)
        x_close = np.concatenate((x_close, np.array(new_df['Close'][i:i+num-1])),axis=0)

        y_train.append([new_df['High'][i + num], new_df['Low'][i + num], new_df['Open'][i + num], new_df['Close'][i + num]])

    y_ = np.array(y_train)
    x_arr = pd.DataFrame(data={'High':x_high,'Low':x_low,'Open':x_open, 'Close':x_close}, columns=['High','Low','Open','Close'])
    y_arr = pd.DataFrame(data={'High':y_[:,0],'Low':y_[:,1],'Open':y_[:,2]}, columns=['High','Low','Open','Close'])

    return x_arr.T,y_arr.T


def five_minutes_split_class(df, num):
    new_df = df.reset_index()
    y_train = []
    x_high = np.zeros(0)
    x_low = np.zeros(0)
    x_open = np.zeros(0)
    x_close = np.zeros(0)

    for i in range(0, (len(df)-len(df)%6)-num):
        x_high = np.concatenate((x_high, np.array(new_df['High'][i:i+num-1])),axis=0)
        x_low = np.concatenate((x_low, np.array(new_df['Low'][i:i+num-1])),axis=0)
        x_open = np.concatenate((x_open, np.array(new_df['Open'][i:i+num-1])),axis=0)
        x_close = np.concatenate((x_close, np.array(new_df['Close'][i:i+num-1])),axis=0)

        y_train.append([new_df['High'][i + num], new_df['Low'][i + num], new_df['Open'][i + num], new_df['Close'][i + num]])

    y_ = np.array(y_train)
    x_arr = pd.DataFrame(data={'High':x_high,'Low':x_low,'Open':x_open, 'Close':x_close}, columns=['High','Low','Open','Close'])
    y_arr = pd.DataFrame(data={'High':y_[:,0],'Low':y_[:,1],'Open':y_[:,2],'Close':y_[:,3]}, columns=['High','Low','Open','Close'])

    return x_arr.T,y_arr.T

def five_minutes_split_gpt_data(df, num):
    # Reshape the input dataframe into a 3D array of shape (N, num, 4), where N is the number of 5-minute intervals in the data
    data = df[['High', 'Low', 'Open', 'Close']].values
    n_intervals = data.shape[0] // num
    data = data[:n_intervals*num].reshape(n_intervals, num, 4)

    # Split the 3D array into two 2D arrays: one for X data and one for Y data
    x_arr = data[:, :-1, :].reshape(n_intervals, -1)
    y_arr = data[:, -1, :]

    # Convert the 2D arrays to pandas DataFrames and transpose them
    x_arr = pd.DataFrame(x_arr.T, columns=['High', 'Low', 'Open', 'Close'])
    y_arr = pd.DataFrame(y_arr.T, columns=['High', 'Low', 'Open', 'Close'])

    return x_arr, y_arr

def normalization(data):
    scaler = MaxAbsScaler()
    scaler_fit = scaler.fit(data)
    result = scaler_fit.transform(data)
    return result, scaler


def normalize_data(data):
    for i  in range (len(data['High'])):
        data['Open'][i] = float(data['Open'][i])/s_price-1
        data['High'][i] = float(data['High'][i])/s_price-1
        data['Low'][i] = float(data['Low'][i])/s_price-1
        data['Close'][i] = float(data['Close'][i])/s_price-1

    return data

def inverse_normalize_data(data):
    for i  in range (len(data)):
        data[i][0] = (float(data[i][0])+1)*s_price
        data[i][1] = (float(data[i][1])+1)*s_price
        data[i][2] = (float(data[i][2])+1)*s_price
        data[i][3] = (float(data[i][3])+1)*s_price
    return data

def gru_model(timesteps, input_dim, output_dim):
    model = Sequential()

    # First GRU layer
    model.add(GRU(units=64, activation='tanh', return_sequences=True, input_shape=(timesteps, input_dim)))

    # Second GRU layer
    model.add(GRU(units=32, activation='tanh', return_sequences=True))

    # Third GRU layer
    model.add(GRU(units=16, activation='tanh', return_sequences=False))

    # Dense hidden layer
    model.add(Dense(units=16, activation='relu'))

    # Output layer
    model.add(Dense(units=output_dim, activation='linear'))

    # Compile the model
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
    return model


def scaleDown(df, name, i):
    return round(df[name].iloc[i]/math.pow(10,len(str(int(df[name].iloc[i])))),3)


def print_epoch_stats(epoch, logs):
    print('Epoch {}: loss = {:.2f}, mae = {:.2f}'.format(epoch, logs['loss'], logs['mean_absolute_error']))


def make_minute_gru_model(train_path, model_path):
    df = read_train_csv_data(train_path)
    nor_df = normalize_data(df)
    num = 31
    x_df, y_df = five_minutes_split_class(nor_df, num)
    print('step1. split data')
    
    #normal_x_df = normalize_data(x_df)
    #normal_y_df = normalize_data(y_df)

    print('step2. normalize data')


    n_x_df = np.swapaxes(x_df.to_numpy(),0,1)
    n_y_df = np.swapaxes(y_df.to_numpy(),0,1)

    temp_mod = int(n_x_df.shape[0]) % (num-1)
    x_train = np.swapaxes(n_x_df[:len(n_x_df)-temp_mod,].reshape(num-1,-1,4),0,1)
    y_train = n_y_df.reshape(-1,4)[:x_train.shape[0]]

    if len(x_train) != len(y_train):
        if len(x_train) > len(y_train):
            x_train = x_train[:len(y_train),]
        else:
            y_train = y_train[:len(x_train), ]
    model = gru_model(len(x_train[9]),4,4)
    print('step3. get gru model')

    model.summary()
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
    train_history = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, verbose=0, callbacks=[tb_callback,LambdaCallback(on_epoch_end=lambda epoch, logs: print_epoch_stats(epoch, logs))])
    print(train_history)
    model.save(model_path)
    print('step4. save model')


def get_date(data_path, num):
    df = read_train_csv_data(data_path)
    nor_df = normalize_data(df)
    x_df, y_df = five_minutes_split_class(nor_df, num)

    n_x_df = np.swapaxes(x_df.to_numpy(),0,1)
    n_y_df = np.swapaxes(y_df.to_numpy(),0,1)

    temp_mod = int(n_x_df.shape[0]) % (num-1)
    x_test = np.swapaxes(n_x_df[:len(n_x_df)-temp_mod,].reshape(num-1,-1,4),0,1)
    y_test = n_y_df.reshape(-1,4)[:x_test.shape[0]]


    if len(x_test[0]) < len(y_test):
        y_test = y_test[:len(x_test[0])]
    else :
        x_test = x_test[:, :len(y_test):]

    x_df.T['Open'] = ((x_df.T['Open'])+1)*s_price
    x_df.T['High'] = ((x_df.T['High'])+1)*s_price
    x_df.T['Low'] = ((x_df.T['Low'])+1)*s_price
    x_df.T['Close'] = ((x_df.T['Close'])+1)*s_price

    y_df.T['Open'] = ((y_df.T['Open'])+1)*s_price
    y_df.T['High'] = ((y_df.T['High'])+1)*s_price
    y_df.T['Low'] = ((y_df.T['Low'])+1)*s_price
    y_df.T['Close'] = ((y_df.T['Close'])+1)*s_price
    return x_test, y_test,x_df.T,y_df.T

def predict_minute_model(train_path, model_path):
    x_test, y_test,x_ori, y_ori = get_date2(train_path,31)
    model = tf.keras.models.load_model(model_path)
    model.summary()

    high_arr = []
    low_arr = []
    open_arr = []
    close_arr = []

    high_arr_t = []
    low_arr_t = []
    open_arr_t = []
    close_arr_t = []

    df_col = ['p_high', 'p_low', 'p_open', 'p_close', 'TP_high', 'TP_low', 'TP_open', 'TP_close', 'diff_high',
              'diff_low', 'diff_open', 'diff_close']
    save_path = 'bit_pred_{}.csv'.format(today)


    y_ori['Open'] = ((y_ori['Open'])+1)*s_price
    y_ori['High'] = ((y_ori['High'])+1)*s_price
    y_ori['Low'] = ((y_ori['Low'])+1)*s_price
    y_ori['Close'] = ((y_ori['Close'])+1)*s_price


    for i in range(len(x_test)):
        xx = np.expand_dims(x_test[i],axis=0)
        yy =np.expand_dims(y_test[i],axis=0)
        test_loss, test_mse = model.evaluate(xx, yy,verbose=0)
        pred = model.predict(xx)
        result = inverse_normalize_data(pred)

        print("%d : test loss = %f, test mse = %f, prediction= [%f, %f, %f], TP = [%f, %f, %f]".format(i, test_loss, test_mse,result[0][0],result[0][1],result[0][2],y_ori['High'][i],y_ori['Low'][i],y_ori['Open'][i]))
        print('test_loss={},test_mse={},prediction=[{},{},{},{}],TP=[{},{},{},{}]'.format(test_loss, test_mse,result[0][0],result[0][1],result[0][2],result[0][3], y_ori['High'][i],y_ori['Low'][i],y_ori['Open'][i],y_ori['Close'][i]))
        
        high_arr.append(result[0][0])
        low_arr.append(result[0][1])
        open_arr.append(result[0][2])
        close_arr.append(result[0][3])

        high_arr_t.append(y_ori['High'][i])
        low_arr_t.append(y_ori['Low'][i])
        open_arr_t.append(y_ori['Open'][i])
        close_arr_t.append(y_ori['Close'][i])
        result_data = {'p_high': result[0][0], 'p_low': result[0][1], 'p_open':result[0][2], 'p_close':result[0][3],
                     'TP_high': y_ori['High'][i], 'TP_low': y_ori['Low'][i], 'TP_open':y_ori['Open'][i], 'TP_close':y_ori['Close'][i],
                     'diff_high':  float(y_ori['High'][i])-float(result[0][0]), 'diff_low': float(y_ori['Low'][i])-float(result[0][1]),
                     'diff_open': float(y_ori['Open'][i])-float(result[0][2]), 'diff_close': float(y_ori['Close'][i])-float(result[0][3])
                     }
        result_df = pd.DataFrame(data=result_data,columns=df_col, index=[0])

        if not os.path.isfile(save_path):
            result_df.to_csv(save_path, mode='w', sep=',', index=False, header=True)
        else :
            result_df.to_csv(save_path, mode='a', sep=',', index=False, header=False)


def make_deep_model_240422():
    train_path = "bitcoin_price_minute_{}.csv".format(today)
    test_path = 'bitcoin_minute_test_data_{}.csv'.format(today)
    model_path ='bitcoin_minute_{}.h5'.format(today)

    make_minute_gru_model(train_path,model_path)
    predict_minute_model(test_path, model_path)

if __name__ == "__main__":
    make_deep_model_240422()