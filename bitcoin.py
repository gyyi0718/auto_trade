import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler
import tensorflow as tf
import math
from datetime import date, timedelta
from keras.optimizers import SGD, Adam
import time
from typing import Union, Optional, List, Dict
from pickle import dump, load
import matplotlib.pyplot as plt


today = datetime.datetime.today().strftime('%y%m%d')
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

def read_csv_data(path):
    train_path = path
    train = pd.read_csv(train_path)
    df = pd.DataFrame(data = {'high':train['high'],'low':train['low'],'open':train['open']}, columns = ['high','low','open'])

    return df



def split_data(df,num):
    column = list(df.keys())
    x_ = pd.DataFrame(df.iloc[0])
    y_ = pd.DataFrame(df.iloc[num])
    #for i in range(len(df)):
    for i in range(0,1000):
        if i % num == 0 and i !=0:
            y_ = pd.concat([y_, df.iloc[i]],ignore_index=True,axis=1)
        elif i%num !=0 and i !=0:
            x_ = pd.concat([x_, df.iloc[i]],ignore_index=True,axis=1)

    return x_.T,y_.T


def temp_split_data(df, num):
    new_df = df.reset_index()
    y_cnt =len(new_df['index'] %num !=0)//num
    x_ = new_df[new_df['index'] %num !=0]
    y_ = new_df[new_df['index'] !=0][new_df['index'] % num==0]
    x_arr = pd.DataFrame(data={'high':x_['high'],'low':x_['low'],'open':x_['open']}, columns=['high','low','open'])
    y_arr = pd.DataFrame(data={'high':y_['high'],'low':y_['low'],'open':y_['open']}, columns=['high','low','open'])

    return x_arr.T,y_arr.T

def normalization(data):
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(data)

    return training_data,scaler


def model_layer(input_shape):
    # h1 ~ h3은 히든 레이어, 층이 깊을 수록 정확도가 높아질 수 있음
    # relu, tanh는 활성화 함수의 종류
    inputs = tf.keras.Input(shape=(input_shape,3))
    h1 = tf.keras.layers.Dense(32, activation='relu')(inputs)
    h2 = tf.keras.layers.Dense(64, activation='tanh')(h1)
    h3 = tf.keras.layers.Dense(128, activation='relu')(h2)
    h4 = tf.keras.layers.Dense(128, activation='tanh')(h3)
    h5 = tf.keras.layers.Dense(128, activation='relu')(h4)

    # 값을 0 ~ 1 사이로 표현할 경우 sigmoid 활성화 함수 활용
    # 마지막 아웃풋 값은 1개여야 함
    h6 = tf.keras.layers.Flatten()(h5)
    h7 = tf.keras.layers.Dense(128, activation='relu')(h6)
    h8 = tf.keras.layers.Dense(64, activation='tanh')(h7)
    h9 = tf.keras.layers.Dense(32, activation='relu')(h8)
    outputs = tf.keras.layers.Dense(3, activation='sigmoid')(h9)

    # 인풋, 아웃풋 설정을 대입하여 모델 생성
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def scaleDown(df, name, i):
    return round(df[name].iloc[i]/math.pow(10,len(str(int(df[name].iloc[i])))),3)

def makeX(df, i):
    tempX = []
    tempX.append(df[1][i])
    tempX.append(df[0][i])

    return tempX

def classification_fun():
    train_path = 'D:/ygy_work/coin/bitcoin.json'
    df = read_json_data(train_path)
    normal_df,scaler = normalization(df)
    model = model_layer(1)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    x = []
    y = []
    # 인풋/아웃풋 데이터 생성

    x = np.array(normal_df[:, 1][:-7])
    y = np.array(normal_df[:, 0][:-7])
    test_x = np.array(normal_df[:, 1][-7:])
    test_y = np.array(normal_df[:, 0][-7:])

    fitInfo = model.fit(x, y, epochs=2000)
    model.save('bitcoine_%s.h5'%today)
    new_model = tf.keras.models.load_model('bitcoine_%s.h5'%today)
    new_model.summary()

    for i in range(len(test_x) - 1):
        test_loss, test_acc = model.evaluate(test_x[i:i + 1], test_y[i:i + 1], verbose=2)
        print("%d : test loss = %f, test acc = %f, [%f]" % (i, test_loss, test_acc, test_y[i:i + 1]))


def make_hour_model():
    train_path = 'D:/ygy_work/coin/bitcoin.csv'
    df = read_csv_data(train_path)
    num = 5
    x_df, y_df = split_data(df,num)

    normal_x_df,x_scaler = normalization(x_df)
    normal_y_df,y_scaler = normalization(y_df)

    temp_mod = int(normal_x_df.shape[0]) % (num-1)
    x_train = normal_x_df[:len(normal_x_df)-temp_mod,].reshape(num-1,-1,3)
    y_train = normal_y_df.reshape(-1,3)

    model = model_layer((num-1))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mse'])


    x_ = x_train[:-(num-1)*10]
    train_x = x_.reshape(-1,(num-1))
    y_ = y_train[:-10]
    test_x = x_[-(num-1)*10:]
    test_x = test_x.reshape((num-1),-1)
    test_y = y_[-10:]
    model.summary()

    train_history = model.fit(train_x, y_, epochs=1000, batch_size=10, validation_split=0.2, verbose=0)
    model.save('bitcoine_hour_%s.h5'%today)
    new_model = tf.keras.models.load_model('bitcoine_hour_%s.h5'%today)
    new_model.summary()

    start_date = date(2018, 1, 1)

    print(test_y)
    for i in range(len(test_y)):
        xx = np.expand_dims(test_x[:,i],axis=0)
        yy =np.expand_dims(test_y[i],axis=0)
        test_loss, test_mse = new_model.evaluate(xx, yy,verbose=0)
        eval_result  = new_model.evaluate(xx, yy,verbose=0)
        result = new_model.predict(xx)
        #print(eval_result)
        print(result)

        print("%d : test loss = %f, test mse = %f, prediction= [%f], TP = [%f]" % (i, test_loss, test_mse,result[0],test_y[i]))



def make_minute_model():
    train_path = 'bitcoin_minute_1126.csv'
    df = read_csv_data(train_path)
    num = 6

    x_df, y_df = temp_split_data(df,num)

    normal_x_df,x_scaler = normalization(x_df)
    normal_y_df,y_scaler = normalization(y_df)

    dump(x_scaler, open('./x_scaler.pkl', 'wb'))
    dump(y_scaler, open('./y_scaler.pkl', 'wb'))


    n_x_df = np.swapaxes(normal_x_df,0,1)
    n_y_df = np.swapaxes(normal_y_df,0,1)

    temp_mod = int(n_x_df.shape[0]) % (num-1)
    x_train = np.swapaxes(n_x_df[:len(n_x_df)-temp_mod,].reshape(num-1,-1,3),0,1)
    y_train = n_y_df.reshape(-1,3)[:x_train.shape[0]]

    if len(x_train) != len(y_train):
        if len(x_train) > len(y_train):
            x_train = x_train[:len(y_train),]
        else:
            y_train = y_train[:len(x_train), ]
    model = model_layer(num-1)
    model = model_layer(num-1)
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mse'])
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
    train_history = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, verbose=0, callbacks=[tb_callback])
    print(train_history)
    model.save('bitcoine_minute_1126.h5')



def get_date(data_path, num):
    df = read_csv_data(data_path)
    x_df, y_df = split_data(df, num)

    normal_x_df, x_scaler = normalization(x_df)
    normal_y_df, y_scaler = normalization(y_df)
    temp_mod = int(normal_x_df.shape[0]) % (num - 1)
    x_test = normal_x_df[:len(normal_x_df) - temp_mod, ].reshape(num - 1, -1, 3)
    y_test = normal_y_df.reshape(-1, 3)
    if len(x_test[0]) < len(y_test):
        y_test = y_test[:len(x_test[0])]
    else :
        x_test = x_test[:, :len(y_test):]

    return x_test, y_test,x_df,y_df

def predict_minute_model(train_path, model_path):
    x_test, y_test,x_ori, y_ori = get_date(train_path,6)
    model = tf.keras.models.load_model(model_path)
    model.summary()



    x_scaler = load(open('x_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))

    high_arr = []
    low_arr = []
    open_arr = []

    high_arr_t = []
    low_arr_t = []
    open_arr_t = []

    y_scaler.fit(y_ori)
    print(y_test)
    for i in range(len(y_test)):
        xx = np.expand_dims(x_test[:,i],axis=0)
        yy =np.expand_dims(y_test[i],axis=0)
        test_loss, test_mse = model.evaluate(xx, yy,verbose=0)
        pred = model.predict(xx)
        result = y_scaler.inverse_transform(pred)
        print("%d : test loss = %f, test mse = %f, prediction= [%f, %f, %f], TP = [%f, %f, %f]"
              % (i, test_loss, test_mse,result[0][0],result[0][1],result[0][2],y_ori['high'][i],y_ori['low'][i],y_ori['open'][i]))
        high_arr.append(result[0][0])
        low_arr.append(result[0][1])
        open_arr.append(result[0][2])

        high_arr_t.append(y_ori['high'][i])
        low_arr_t.append(y_ori['low'][i])
        open_arr_t.append(y_ori['open'][i])

    plt.figure(figsize=(20,5))
    plt.plot(high_arr, color='r')
    plt.plot(high_arr_t, color='magenta',linestyle='dashed',alpha=0.5)

    plt.plot(low_arr, color='g')
    plt.plot(low_arr_t, color='limegreen',linestyle='dashed',alpha=0.5)

    plt.plot(open_arr, color='b')
    plt.plot(open_arr_t, color='aqua',linestyle='dashed',alpha=0.5)

    plt.savefig('predict.png')


if __name__ == "__main__":
    #make_minute_model()
    train_path = 'bitcoin_minute_221127_USD.csv'
    model_path ='bitcoine_minute_1126.h5'
    predict_minute_model(train_path, model_path)