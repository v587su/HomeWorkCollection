import pandas as pd
import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TRAIL_INTERVAL = 500
TRAIL_LENGTH = 6
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
L2_NORM = 0.001
MIN_LON = 121.20120485
MIN_LAT = 31.28175691
MAX_LON = 121.21831882
MAX_LAT = 31.29339344


class Trails(Dataset):
    def __init__(self, d, t, m, train=True):
        super(Trails, self).__init__()
        label_list = ['Longitude', 'Latitude']

        dd = d.copy()
        mm = m.copy()

        # 对数据进行归一化处理
        if train:
            dd['Longitude'] = dd['Longitude'].sub(MIN_LON).div(MAX_LON - MIN_LON)
            dd['Latitude'] = dd['Latitude'].sub(MIN_LAT).div(MAX_LAT - MIN_LAT)
            self.tgt = dd[label_list].values

        mm = mm.values.reshape(-1, 5)
        for ii in range(5):
            x = mm[:, ii]
            mm[:, ii] = (x - np.mean(x)) / np.std(x)
        mm = mm.reshape(-1, 6, 5)
        sp = dd['Speed'].values
        self.mat = mm
        self.tra = t
        self.speed = (sp - np.mean(sp)) / np.std(sp)
        self.train = train

    def __getitem__(self, idx):
        tra = self.tra[idx]
        tras = [int(a) for a in tra]
        mat = self.mat[tras]
        if self.train:
            tgt = self.tgt[tras]
        else:
            tgt = tra
        speed = self.speed[tras]
        return mat, tgt, speed

    def __len__(self):
        return len(self.tra)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5,
                      padding=2),

            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),
        )
        self.linear = nn.Sequential(
            nn.Linear(12, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU()
        )

        normalize_layer = [0, 3, 6, 11, 14, 17]
        # normalize_layer = [0, 3, 8, 11]
        for i in normalize_layer:
            nn.init.kaiming_normal_(self.model[i].weight,
                                    nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear[0].weight,
                                nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear[2].weight,
                                nonlinearity='leaky_relu')

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), output.size(1), -1)
        output = self.linear(output)
        return output


class CLSTM(nn.Module):
    def __init__(self):
        super(CLSTM, self).__init__()
        self.cnn = CNN()
        # self.lstm = nn.LSTM(input_size=32, hidden_size=1,
        #                     batch_first=True, bidirectional=True)
        self.lstm = nn.GRU(input_size=32, hidden_size=1,
                           batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(17, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU()
        )

    def forward(self, m, speed):
        # 先过cnn，出来的结果concat speed然后喂进lstm
        fea = self.cnn(m)
        speed_size = speed.size()
        fea_speed = torch.cat((fea, speed.view(speed_size[0], -1, 1)),
                              dim=2)
        fea_speed = self.linear(fea_speed)
        result, _ = self.lstm(fea_speed)
        return result


def clean_matrix(matrix):
    last_available = None
    for row in matrix.iterrows():
        if pd.isnull(row[1]['Latitude']):
            matrix.loc[row[0]] = last_available
        else:
            last_available = row[1]
    return matrix


def get_matrix(row, station):
    matrix = np.zeros((6, 5))
    feature_list = ['RNCID', 'CellID', 'Dbm', 'AsuLevel', 'SignalLevel']
    for i in range(1, 7):
        for j in range(len(feature_list)):
            column_name = '{}_{}'.format(feature_list[j], str(i))
            matrix[i - 1, j] = row[1][column_name]
    matrix = pd.DataFrame(matrix, columns=feature_list).join(station,
                                                             on=['RNCID',
                                                                 'CellID'])
    matrix = clean_matrix(matrix)
    # matrix['Longitude'] = matrix['Longitude'].sub(MIN_LON).div(
    #     MAX_LON - MIN_LON)
    # matrix['Latitude'] = matrix['Latitude'].sub(MIN_LAT).div(
    #     MAX_LAT - MIN_LAT)
    matrix = matrix[matrix.columns.difference(['RNCID', 'CellID'])]
    return matrix


def get_trail(data):
    trails = []
    for i in data.groupby('TrajID'):
        my_trail = i[1].index.values.tolist()
        my_trail = [str(a) for a in my_trail]
        trails.append(my_trail)
    trails = split_trail(trails)
    return trails


def split_trail(trails):
    trails_splited = []
    for i, trail in enumerate(trails):
        while len(trail) > TRAIL_LENGTH:
            trails_splited.append(trail[:TRAIL_LENGTH])
            trail = trail[TRAIL_LENGTH:]
        if len(trail) < TRAIL_LENGTH:
            trail = padding_trail(trail, trails_splited[-1])
        trails_splited.append(trail)
    return trails_splited


def padding_trail(trail, last_trail):
    padding_num = TRAIL_LENGTH - len(trail)
    trail = last_trail[-padding_num:] + trail
    return trail
