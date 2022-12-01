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
from pickle import dump,load
import os
from mpl_finance import candlestick2_ohlc
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


def split_dataframe(df, num):
    new_df = df.reset_index()
    y_cnt =len(new_df['index'] %num !=0)//num
    x_ = new_df[new_df['index'] %num !=0]
    y_ = new_df[new_df['index'] !=0][new_df['index'] % num==0]
    x_arr = pd.DataFrame(data={'high':x_['high'],'low':x_['low'],'open':x_['open']}, columns=['high','low','open'])
    y_arr = pd.DataFrame(data={'high':y_['high'],'low':y_['low'],'open':y_['open']}, columns=['high','low','open'])

    return x_arr.T,y_arr.T

def normalization(data):
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(np.expand_dims(np.array(data),axis=1))
    return training_data, scaler

def invert_normalization(scaler, data):
    test_data = scaler.inverse_transform(np.expand_dims(np.expand_dims(data,axis=0),axis=1))
    return test_data[0][0]


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
    h10 = tf.keras.layers.Dense(3, activation='sigmoid')(h9)
    outputs = tf.keras.layers.Activation('softmax')(h10)

    # 인풋, 아웃풋 설정을 대입하여 모델 생성
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def model_layer2(input_shape):
    # h1 ~ h3은 히든 레이어, 층이 깊을 수록 정확도가 높아질 수 있음
    # relu, tanh는 활성화 함수의 종류
    inputs = tf.keras.Input(shape=(input_shape,))
    h1 = tf.keras.layers.Dense(32, activation='relu')(inputs)
    h2 = tf.keras.layers.Dense(64, activation='tanh')(h1)
    h3 = tf.keras.layers.Dense(128, activation='relu')(h2)
    h4 = tf.keras.layers.Dense(64, activation='tanh')(h3)
    h5 = tf.keras.layers.Dense(32, activation='relu')(h4)
    h6 = tf.keras.layers.Dense(1)(h5)
    #outputs = tf.keras.layers.Activation('softmax')(h6)

    # 인풋, 아웃풋 설정을 대입하여 모델 생성
    model = tf.keras.Model(inputs=inputs, outputs=h6)
    return model


def scaleDown(df, name, i):
    return round(df[name].iloc[i]/math.pow(10,len(str(int(df[name].iloc[i])))),3)

def makeX(df, i):
    tempX = []
    tempX.append(df[1][i])
    tempX.append(df[0][i])

    return tempX

def classification_fun():
    train_path = 'D:/ygy_work/coin/b.json'
    df = read_json_data(train_path)
    normal_df, scaler = normalization(df)

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
    model.save('b_%s.h5'%today)
    new_model = tf.keras.models.load_model('b_%s.h5'%today)
    new_model.summary()

    for i in range(len(test_x) - 1):
        test_loss, test_acc = model.evaluate(test_x[i:i + 1], test_y[i:i + 1], verbose=2)
        print("%d : test loss = %f, test acc = %f, [%f]" % (i, test_loss, test_acc, test_y[i:i + 1]))


def make_hour_model():
    train_path = 'D:/ygy_work/time_series/b.csv'
    df = read_csv_data(train_path)
    num = 5
    x_df, y_df = split_data(df,num)

    normal_x_df, scaler = normalization(x_df)
    normal_y_df, scaler = normalization(y_df)

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
    model.save('b_hour_%s.h5'%today)
    new_model = tf.keras.models.load_model('b_hour_%s.h5'%today)
    new_model.summary()

    start_date = date(2018, 1, 1)

    print(test_y)
    for i in range(len(test_y)):
        xx = np.expand_dims(test_x[i],axis=0)
        yy =np.expand_dims(test_y[i],axis=0)
        test_loss, test_mse = new_model.evaluate(xx, yy,verbose=0)
        pred_data = new_model.predict(xx)
        pred_result = invert_normalization(np.array(pred_data[0]))
        pred_arr = [abs(pred_result[0]-test_y[i][0]),abs(pred_result[1]-test_y[i][1]),abs(pred_result[2]-test_y[i][2])]
        print("mse = %f , %f, %f"%(pred_arr[0], pred_arr[1], pred_arr[2]))
        print("%d : test loss = %f, test mse = %f, prediction= [%f], TP = [%f]" % (i, test_loss, test_mse,pred_result,test_y[i]))



def make_minute_model():
    train_path = 'b_minute_221127.csv'
    df = read_csv_data(train_path)
    num = 6
    x_df, y_df = split_dataframe(df,num)
    normal_x_df = []
    normal_y_df = []
    for i in df.keys():
        temp_x, scaler_x = normalization(x_df.T[i])
        temp_y, scaler_y = normalization(y_df.T[i])

        dump(scaler_x, open('./x_scaler_%s.pkl'%i, 'wb'))
        dump(scaler_y, open('./y_scaler_%s.pkl'%i, 'wb'))

        normal_x_df.append(temp_x[:,])
        normal_y_df.append(temp_y[:,])


    n_x_df = np.swapaxes(np.array(normal_x_df[2]),0,1)
    n_y_df = np.swapaxes(np.array(normal_y_df[2]),0,1)

    temp_mod = int(n_x_df.shape[0]) % (num-1)
    x_train = np.swapaxes(n_x_df[:len(n_x_df)-temp_mod,].reshape(num-1,-1,3),0,1)
    y_train = n_y_df.reshape(-1,3)[:x_train.shape[0]]

    #model = model_layer(num-1)
    model = model_layer2(num-1)

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mse'])
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1)
    train_history = model.fit(x_train[:,:,2], y_train[:,2], epochs=1000, batch_size=10, validation_split=0.2, verbose=0, callbacks=[tb_callback])
    print(train_history)
    model.save('b_minute[open]_%s.h5'%(today))

def make_minute_model_one():
    train_path = 'b_minute_221127.csv'
    df = read_csv_data(train_path)
    num = 6
    x_df, y_df = split_dataframe(df,num)
    normal_x_df = []
    normal_y_df = []
    for i in df.keys():
        temp_x, scaler_x = normalization(x_df.T[i])
        temp_y, scaler_y = normalization(y_df.T[i])

        dump(scaler_x, open('./x_scaler_%s.pkl'%i, 'wb'))
        dump(scaler_y, open('./y_scaler_%s.pkl'%i, 'wb'))

        normal_x_df.append(temp_x[:,])
        normal_y_df.append(temp_y[:,])

    normal_y_df = normal_y_df[2][:,0]
    normal_x_df = normal_x_df[0][:,0]

    n_x_df = normal_x_df.reshape(5,-1)
    n_y_df = normal_y_df.reshape(1,-1)

    temp_mod = int(n_x_df.shape[0]) % (num-1)
    x_train = np.swapaxes(n_x_df,0,1)
    y_train = n_y_df.reshape(-1,1)

    if len(x_train)> len(y_train):
        x_train = x_train[:len(y_train)]
    else:
        y_train = y_train[:len(x_train)]

    #model = model_layer(num-1)
    model = model_layer2(num-1)

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mse'])
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1)
    train_history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2, verbose=0, callbacks=[tb_callback])
    print(train_history)
    model.save('b_minute[open]_%s.h5'%(today))


def get_date(data_path, num):
    df = read_csv_data(data_path)
    x_df, y_df = split_dataframe(df, num)

    normal_x_df = []
    normal_y_df = []
    for i in df.keys():
        temp_x, _ = normalization(x_df.T[i])
        temp_y, __ = normalization(y_df.T[i])

        normal_x_df.append(temp_x[:,])
        normal_y_df.append(temp_y[:,])


    n_x_df = np.swapaxes(normal_x_df, 0, 1)
    n_y_df = np.swapaxes(normal_y_df, 0, 1)

    temp_mod = int(n_x_df.shape[0]) % (num - 1)
    x_train = np.swapaxes(n_x_df[:len(n_x_df) - temp_mod, ].reshape(num - 1, -1, 3), 0, 1)
    y_train = n_y_df.reshape(-1, 3)[:x_train.shape[0]]

    if len(x_train) > len(y_train):
        x_train =x_train[:len(y_train)]
    else:
        y_train =y_train[:len(x_train)]

    return x_train, y_train,_ , __

def get_pred_date(data_path, num):
    df = read_csv_data(data_path)
    x_df, y_df = split_dataframe(df, num)

    x_scaler = load(open('./x_scaler_open.pkl', 'rb'))

    normal_x_df = []
    normal_y_df = []

    x_open = np.expand_dims(np.array(x_df.T['open']),axis=1)
    y_open = np.expand_dims(np.array(y_df.T['open']),axis=1)

    temp_mod = int(len(x_open)%(num-1))
    x_open = x_open[:(len(x_open) - temp_mod)]
    temp_x = x_scaler.fit_transform(x_open)
    temp_y = x_scaler.fit_transform(y_open)
    x_train =temp_x.reshape(len(x_open)//(num-1),(num-1),1)
    y_train = temp_y


    if len(x_train) > len(y_train):
        x_train =x_train[:len(y_train)]
    else:
        y_train =y_train[:len(x_train)]

    return x_train, y_train, x_scaler


def predict_minute_model(train_path, model_path):
    x_test, y_test,x_scaler,y_scaler = get_pred_date(train_path,6)
    df = read_csv_data(train_path)
    x_df, y_df = split_dataframe(df, 6)
    model = tf.keras.models.load_model(model_path)
    model.summary()
    y_df_ = y_df.T

    print(y_test)
    s_arr =[]
    s_arr2 = []

    for i in range(len(y_test)):
        xx = np.expand_dims(x_test[i],axis=0)
        yy = np.expand_dims(y_test[i],axis=0)
        test_loss, test_mse = model.evaluate(xx, yy,verbose=0)
        pred_data = model.predict(xx)
        pred1 = s_arr[0].inverse_transform(pred_data[0][0])
        pred2 = s_arr[1].inverse_transform(pred_data[0][1])
        pred3 = s_arr[2].inverse_transform(pred_data[0][2])

        pred_arr = [abs(pred1-y_df_.iloc[i]['high']),abs(pred2-y_df_.iloc[i]['low']),abs(pred3-y_df_.iloc[i]['open'])]
        print("mse = %f , %f, %f"%(pred_arr[0], pred_arr[1], pred_arr[2]))
        print("%d : test loss = %f, test mse = %f, prediction= [%f, %f, %f], TP = [%f, %f, %f]" %
              (i, test_loss, test_mse,pred1,pred2,pred3,y_df_.iloc[i]['high'],y_df_.iloc[i]['low'],y_df_.iloc[i]['open']))

def predict_minute_model2(train_path, model_path):
    x_test, y_test,x_scaler = get_pred_date(train_path,6)
    df = read_csv_data(train_path)
    x_df, y_df = split_dataframe(df, 6)
    model = tf.keras.models.load_model(model_path)
    model.summary()
    mod = len( np.array(x_df)[2])%5
    y_ = np.array(y_test)
    x_ = np.array(x_test)

    pred_arr = []
    tp_arr = []

    print(y_test)
    s_arr =[]
    s_arr2 = []
    save_path = 'b_221129_result3.csv'
    for i in range(len(y_test)):
        xx = np.expand_dims(x_[i],axis=0)
        yy = np.expand_dims(y_[i],axis=0)
        test_loss, test_mse = model.evaluate(xx, yy,verbose=0)
        pred_data = model.predict(xx[:,:,0])
        #print('xx : %s, pred = %s'%(xx[:,:,0], pred_data))
        pred = x_scaler.inverse_transform(pred_data)[0][0]
        tp = np.array(y_df.T['open'])[i]
        pred_mse = abs(pred-tp)
        #print("pred = %f, tp = %f"%(pred,tp))
        save_df = pd.DataFrame(data={'loss':test_loss,'diff':pred_mse,'prediction':pred, 'tp':tp}, columns=['loss','diff','prediction','tp'],index=[0])
        if not os.path.isfile(save_path):
            save_df.to_csv(save_path, mode='w', sep=',', index=False, header=True)
        else:
            save_df.to_csv(save_path, mode='a', sep=',', index=False, header=False)

        print("%d : test loss = %f, diff = %f, prediction= [%f], TP = [%f], %f " %(i, test_loss, pred_mse,pred,tp, pred_data))

        pred_arr.append(pred)
        tp_arr.append(tp)

    plt.plot(tp_arr, color='green')
    plt.plot(pred_arr,color='blue')
    plt.savefig('b_minute_open_pred.png')

def draw_plot(df):
    tp_arr = np.array(df['tp'])
    pred_arr = np.array(df['prediction'])
    start_time = 0
    end_time = len(tp_arr)
    x = np.linspace(0,end_time,end_time)
    x = np.linspace(0,200,200)
    color_fuc = lambda x: 'r' if x >= 0 else 'b'

    for i in range(start_time,end_time,200):


        draw_tp = tp_arr[i:i+200]
        pred_tp = pred_arr[i:i+200]
        diff = pred_tp - draw_tp


        plt.figure(figsize=(25,5))
        below_threshold = diff < 0
        plt.scatter(x[below_threshold], diff[below_threshold], color='blue')
        # Add above threshold markers
        above_threshold = np.logical_not(below_threshold)  # true/false 반대로
        plt.scatter(x[above_threshold], diff[above_threshold], color='red')

        plt.savefig('b_minute_open_pred_diff_%d.png'%i)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model


def bulid_resnet(input_shape, output_shape):
    base = ResNet50(input_shape=5, include_top=False)
    x = Flatten()(base.output)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.inputs, outputs=x)
    return model


def make_minute_resnet_model():
    train_path = 'b_minute_221127.csv'
    ori_df = read_csv_data(train_path)
    new_df = ori_df.diff(1)[1:]
    num = 6

    scaler = MinMaxScaler()
    scale_npy = scaler.fit_transform(np.array(new_df))
    scale_df = pd.DataFrame(data={'high':scale_npy[:,0],'low':scale_npy[:,0],'open':scale_npy[:,2]}, columns=['high','low','open'])
    x_df, y_df = split_dataframe(scale_df, num)
    normal_x_df = []
    normal_y_df = []
    dump(scaler, open('./resnet_scaler_.pkl', 'wb'))


    temp_x  = scaler.fit_transform(x_df)
    temp_y  = scaler.fit_transform(y_df)

    mod = len(temp_x[0])%224
    x_train = temp_x[:,:-mod].reshape(-1,224,224,3)
    y_train = np.swapaxes(temp_y[:,], 0, 1)


    model = bulid_resnet(5,1)
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mse'])
    model.summary()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1)
    train_history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2, verbose=0,
                              callbacks=[tb_callback])
    print(train_history)
    model.save('b_minute_resnet_[open]_%s.h5' % (today))

if __name__ == "__main__":
    #make_minute_model_one()
    train_path = 'b_minute_test.csv'
    model_path ='b_minute[open]_221130.h5'
    #predict_minute_model2(train_path, model_path)
    make_minute_resnet_model()
    df = pd.read_csv('b_221129_result3.csv')
    #draw_plot(df)