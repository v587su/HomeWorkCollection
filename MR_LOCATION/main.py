import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as data
import pandas as pd
import numpy as np
import os.path as path
from util import *

TRAIL_LENGTH_MAX = 30


class Trails(data.Dataset):
    def __init__(self, data_path, tra_path):
        super(Trails, self).__init__()
        d = pd.read_csv(data_path)
        tra = pd.read_csv(tra_path)
        column_list = ['RNCID', 'CellID', 'Dbm', 'AsuLevel', 'SignalLevel']
        label_list = ['Longitude', 'Latitude']
        feature_list = []
        for i in range(1, 7):
            for j in range(len(column_list)):
                column_name = '{}_{}'.format(column_list[j], str(i))
                feature_list.append(column_name)

        self.mat = d[column_list].values
        self.tra = tra.values
        self.tgt = d[label_list].values

    def __getitem__(self, idx):
        tra = self.tra[idx, :]
        mat = self.mat[tra]
        tgt = self.tgt[tra]
        return mat, tgt

    def __len__(self):
        return len(self.tra)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,
                      padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), -1)
        return output


class CLSTM(nn.Module):
    def __init__(self):
        super(CLSTM, self).__init__()
        self.cnn = CNN()
        # 这里的参数待定，3+n,n是cnn出来的长度
        self.linear = nn.Linear(10, 7)
        self.cell_list = nn.ModuleList(
            [nn.LSTMCell(30, 30) for _ in range(TRAIL_LENGTH_MAX)]
        )

    def forward(self, mat_list):
        # todo 写模型，先过cnn，出来的结果concat speed然后喂进lstm
        for


crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
trail_path = path.join(crt_path, 'data_file', 'trail.csv')

a = MyTrails(data_path, trail_path)
