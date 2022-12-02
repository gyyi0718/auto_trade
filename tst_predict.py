import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

input_window = 60  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
batch_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def split_data(df, num):
    new_df = df.reset_index()
    y_cnt = len(new_df['index'] % num != 0) // num
    x_ = new_df[new_df['index'] % num != 0]
    y_ = new_df[new_df['index'] != 0][new_df['index'] % num == 0]
    x_arr = pd.DataFrame(data={'high': x_['high'], 'low': x_['low'], 'open': x_['open']},
                         columns=['high', 'low', 'open'])
    y_arr = pd.DataFrame(data={'high': y_['high'], 'low': y_['low'], 'open': y_['open']},
                         columns=['high', 'low', 'open'])

    return x_arr.T, y_arr.T


def normalization(data):
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(data)
    return training_data


def read_csv_data(path):
    train_path = path
    train = pd.read_csv(train_path)
    df = pd.DataFrame(data={'high': train['high'], 'low': train['low'], 'open': train['open']},
                      columns=['high', 'low', 'open'])

    return df


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def main_fun():
    series = read_csv('bitcoin_minute_test.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(series['open'].to_numpy().reshape(-1, 1)).reshape(-1)
    amplitude = np.array(amplitude, float)
    test_sequence = create_inout_sequences(amplitude, input_window)
    mod = len(amplitude) % 60
    size = len(amplitude) - mod
    test_data = amplitude[:size].reshape(-1, 60)
    model = TransAm().to(device)
    model = torch.load('best_tst_model.pt')
    model.eval()
    for i in test_data:
        df = pd.DataFrame(data={'ori': i}, columns=['ori'])
        df.to_csv('model_ori.csv', mode='a', header=False)
    for i in test_data:
        data1 = np.resize(i, (60, 250))
        data2 = torch.from_numpy(data1).to(device).float()
        out = model(data2)
        out_cpu = out.cpu()
        out_npy = out_cpu.detach().numpy()
        result = scaler.inverse_transform(out_npy[:, :, 0])
        df = pd.DataFrame(data={'result': result.reshape(-1)}, columns=['result'])
        df.to_csv('model_result.csv', mode='a', header=False)


if __name__ == "__main__":
    main_fun()
