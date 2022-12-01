import sys
import os
import h5py
import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tsai.inference import load_learner
from tsai.imports import create_scripts
from tsai.export import get_nb_name
from tsai.data.tabular import *
from tsai.all import *
from tsai.models import *
from fastai.metrics import F1Score
from fastai.basics import *
from fastcore.test import test_eq
import data.data_info as d_obj
import data.dof_train_data as df_obj
import data.draw_plot as draw
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split
from pickle import dump,load
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime

today = datetime.datetime.today().strftime('%y%m%d')

sys.path.append('../../')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Transformer Time Seriese
class TTS():
    def __init__(self):
        self.sequence_len = 1000
        self.input_window = 100
        self.epoch_ = 1000
        self.column = ['high', 'low', 'open']
        self.d_info = d_obj.DofData()
        self.dof_obj = df_obj.DofTrainData(self.d_info)
        self.d_plot = draw.DrawPlot(self.d_info)
        self.cur_dir = self.d_info.cur_dir
        self.train_dir = ''
        self.val_dir = ''

    def get_size(self, line):
        line = line.replace(' \n', '')
        line = line.replace('\n', '')
        line = line.replace('\t', ' ')
        line = line.replace(' ', ' ')
        line = line.split(' ')
        arr_size = len(line)
        return arr_size

    def read_motion_code_file(self, path):
        temp_f = open(path, 'r')
        data_array = []
        temp_line = temp_f.readline()
        arr_size = self.get_size(temp_line)
        temp_f.close()
        f = open(path, 'r')
        if arr_size > 6:
            arr_size = 6
        for i in range(arr_size):
            data_array.append([])
        while 1:
            line = f.readline()
            if not line:
                break
            line = line.replace(',', '')
            line = line.replace(' \n', '')
            line = line.replace('\n', '')
            line = line.replace('\t', ' ')
            line = line.replace(' ', ' ')
            line = line.split(' ')
            line = [n for n in line if n]
            for idx in range(0, arr_size):
                data_array[idx].append(line[idx])
        f.close()
        # print(path)

        for n in range(0, len(data_array)):
            data_array[n] = np.asarray(data_array[n]).astype(float)
        return data_array

    def get_read_file(self, path):
        train_path = path
        train = pd.read_csv(train_path)
        df = pd.DataFrame(data={'high': train['high'], 'low': train['low'], 'open': train['open']},
                          columns=['high', 'low', 'open'])

        return df



    def split_data(self, df, num):
        new_df = df.reset_index()
        y_cnt = len(new_df['index'] % num != 0) // num
        x_ = new_df[new_df['index'] % num != 0]
        y_ = new_df[new_df['index'] != 0][new_df['index'] % num == 0]
        x_arr = pd.DataFrame(data={'high': x_['high'], 'low': x_['low'], 'open': x_['open']},
                             columns=['high', 'low', 'open'])
        y_arr = pd.DataFrame(data={'high': y_['high'], 'low': y_['low'], 'open': y_['open']},
                             columns=['high', 'low', 'open'])

        return x_arr.T, y_arr.T

    def train_data(self, X, Y, model_name):
        if len(X.shape) == 3:
            X = np.swapaxes(X, 0, 1)
        else:
            X = np.expand_dims(X, axis=-1)
            X = np.swapaxes(X, 1, 2)
        split_index = int(len(X) * 0.1)
        splits = get_predefined_splits(X[:-split_index], X[-split_index:])
        # train_model = self.tst_forecast(X[splits[0]], Y[splits[0]], model_name)
        train_model2 = self.tst_train(X, Y, splits, model_name)

    def tst_train(self, X, Y, splits, model_name):
        check_data(X, Y, splits)
        tfms = [None, [TSRegression()]]
        batch_tfms = TSStandardize(by_sample=True, by_var=True)

        dsid = 'AppliancesEnergy'
        x, y, split = get_regression_data(dsid, split_data=False)
        dls = get_ts_dls(x, y, splits=split, tfms=tfms, batch_tfms=batch_tfms, bs=16)
        dls.one_batch()
        dls.show_batch()

        learn = tsai.learner.ts_learner(dls, InceptionTime, metrics=[mae, rmse], cbs=ShowGraph())
        learn.fit_one_cycle(50, 1e-2)
        learn.export("../%s_models/%s.pkl" % (self.d_info.today, model_name))

    def tst_forecast(self, X, Y, model_name):
        # batch_tfms = TSStandardize(by_sample=False)
        batch_noraml = TSNormalize(range(-1, 1), by_sample=True)
        # reg = TSForecaster(X, Y, path='../models', arch=TSTPlus, batch_tfms=batch_tfms, metrics=[mse, rmse],cbs=ShowGraph())
        reg = TSRegressor(X, Y, path='../models', arch=TSTPlus, batch_tfms=batch_noraml, metrics=rmse)
        # cbs=ShowGraph())

        reg.show_training_loop()
        reg.fit_one_cycle(self.epoch_, 1e-4)
        if not os.path.isdir("../%s_models" % self.d_info.today):
            os.mkdir("../%s_models" % self.d_info.today)
        reg.export("../%s_models/%s.pkl" % (self.d_info.today, model_name))

    def model_performance(self):
        model_dir = '/home/user/Project/MCI/result/02_[tst]deep_learning'
        root_dir = '%s/data/MC_Dataset/Film_data_hdf_Test/' % self.cur_dir
        predict_arr = []
        for dof in self.dof_arr:
            self.dof = dof
            model_name = "[%s][%s][%s]_%ddof" % (self.data_type, self.gt_type, dof, self.dof_channel)
            x_train, y_train = self.get_hdf_data(root_dir, dof)
            train_model = load_learner("../%s_models/%s.pkl" % (self.d_info.today, model_name))
            if self.dof_channel == 3:
                train_data = np.swapaxes(x_train, 0, 1)
                predict = self.load_model_and_performance_test(train_model, model_name, train_data, y_train)
            elif self.dof_channel == 1:
                train_data = x_train[self.dof_dic[dof]]
                train_data = np.expand_dims(train_data, axis=-1)
                train_data = np.swapaxes(train_data, 1, 2)
                predict = self.load_model_and_performance_test(train_model, model_name, train_data, y_train)
            predict_arr.append(predict)
            result = self.dof_obj.make_peak_spline_data(predict_arr[0])
        return result

    def model_prediction(self, model_path, test_dir):
        predict_arr = []

        x_ori, y_ori = self.get_read_xtt_files(root_dir=test_dir, type='test', seq=1000)
        x_test, y_test = self.get_train_data(x_ori, y_ori)
        x_normal = tst.normalize_data2(x_test)

        train_model = load_learner(model_path)
        X = np.swapaxes(x_normal, 0, 1)
        Y = np.expand_dims(y_test, axis=0)
        save_dir = '/home/user/Project/MCI/result/221116/'
        raw_preds, target, preds = train_model.get_X_preds(X, Y)
        result_arr = []
        for i in range(len(preds)):
            self.d_plot.draw_original_data_and_predict_plot_fun(X[i], Y[0][i], np.array(preds[i]), save_dir, '%04d' % i,
                                                                ['X', 'Y', 'preds'])

    def load_model_and_performance_test(self, tst, model_name, X, Y):
        preds_arr = np.zeros(Y.shape)
        raw_preds, target, preds = tst.get_X_preds(X, Y)
        result_arr = []

        model_dir = "../../result/%s/[%ddof][%s][%s][%s]" % (
        self.d_plot.today, self.dof_channel, self.dof, self.data_type, self.gt_type)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        if self.dof_channel == 3:
            title_arr = ['roll', 'pitch', 'heave']
            ledends = ['prediction', 'roll', 'pitch', 'heave', 'gt']
        elif self.dof_channel == 1:
            title_arr = [self.dof]
            ledends = ['prediction', '%s' % self.dof, 'gt']
        preds_np = self.dof_obj.make_peak_spline_data(np.array(raw_preds))
        for i in range(len(preds_np)):
            # filter = self.dof_obj.temp_lowpass_filter(preds_np[i])
            preds_arr[i] = preds_np[i]
            # data_arr = [np.expand_dims(preds_arr[i],axis=0),np.swapaxes(X[i],0,1), np.expand_dims(Y[i],axis=0)]
            data_arr = [np.expand_dims(preds_arr[i], axis=0), X[i], np.expand_dims(Y[i], axis=0)]

            self.d_plot.draw_compare_data_plots(data_arr, "%s/%04d_" % (model_dir, i), title_arr, ledends)
            self.dof_obj.performance_metric(preds_arr[i], np.array(target[i]), model_dir, model_name)
        data_arr = [np.expand_dims(preds_arr, axis=0), np.swapaxes(np.swapaxes(X, 2, 0), 1, 2),
                    np.expand_dims(Y, axis=0)]
        # self.d_plot.draw_sliding_plots(data_arr, "%s/" % model_dir, title_arr, ledends)
        return preds_arr

    def normalize_data(self, data):
        scaler = MinMaxScaler()
        training_data = scaler.fit_transform(np.expand_dims(np.array(data), axis=1))
        return training_data, scaler

    def invert_normalization(self, scaler, data):
        test_data = scaler.inverse_transform(np.expand_dims(np.expand_dims(data, axis=0), axis=1))
        return test_data[0][0]
if __name__ == "__main__":
    tst = TTS()
    tst.sequence_len = 3600
    tst.epoch_ = 1000
    tst.val_dir = 'bitcoin_minute_test.csv'
    tst.train_dir = 'bitcoin_minute.csv'
    root_dir = '/home/user/Project/MCI/data/221111_deep_data/'
    df = tst.get_read_file(tst.train_dir)
    x_train, y_train = tst.split_data(df, tst.sequence_len)
    x_normal,scaler_x = tst.normalize_data(x_train)
    y_normal,scaler_y = tst.normalize_data(y_train)

    dump(scaler_x, open('./x_scaler_%s.pkl' % tst.column[2], 'wb'))
    dump(scaler_y, open('./y_scaler_%s.pkl' % tst.column[2], 'wb'))

    X, Y, preds = tst.train_data(x_normal, y_normal, 'bitcoin_tst_%s'%today)
    deep_path = '../bitcoin_tst_%s.pkl'%today
    tst.model_prediction(model_path=deep_path, test_dir=root_dir)