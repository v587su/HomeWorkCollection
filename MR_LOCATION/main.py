import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os.path as path
from util import *


class Trails(Dataset):
    def __init__(self, d, t):
        super(Trails, self).__init__()
        column_list = ['RNCID', 'CellID', 'Dbm', 'AsuLevel', 'SignalLevel']
        label_list = ['Longitude', 'Latitude']
        feature_list = []
        for i in range(1, 7):
            for j in range(len(column_list)):
                column_name = '{}_{}'.format(column_list[j], str(i))
                feature_list.append(column_name)


        self.mat = d[feature_list].values
        self.tra = t
        self.tgt = d[label_list].values
        self.speed = d['Speed'].values

    def __getitem__(self, idx):
        tra = self.tra[idx]
        tra = [int(a) for a in tra]
        mat = self.mat[tra]
        tgt = self.tgt[tra]
        speed = self.speed[tra]
        return mat, tgt, speed

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
    def __init__(self, rnn_length):
        super(CLSTM, self).__init__()
        self.cnn = CNN()
        # 这里的参数待定，3+n,n是cnn出来的长度
        self.linear = nn.Linear(10, 7)
        self.lstm = nn.LSTM(input_size=rnn_length, hidden_size=20)

    def forward(self, mat_list, speed):
        # 先过cnn，出来的结果concat speed然后喂进lstm
        fea_list = []
        for mat in mat_list:
            fea = self.cnn(mat)
            fea = torch.cat((fea, speed), dim=1)
            fea_list.append(fea)

        fea_ts = torch.stack(fea_list, dim=0)
        result, _ = self.lstm(fea_ts)
        return result


crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
trail_path = path.join(crt_path, 'data_file', 'trail.csv')
data = pd.read_csv(data_path)
f = open(trail_path, 'r')
trail = f.readlines()
trail = [a.strip('\n').split(',') for a in trail]
f.close()
train_idx, val_idx = train_test_split(trail, test_size=0.2, shuffle=True)

train_data = DataLoader(Trails(data, train_idx), batch_size=BATCH_SIZE)
val_data = DataLoader(Trails(data, val_idx), batch_size=BATCH_SIZE)

model = CLSTM(TRAIL_LENGTH)
loss = nn.MSELoss()
if USE_CUDA:
    model.cuda()
    loss.cuda()
optimizer = optim.Adam(model.parameters())

steps = len(train_data)
for epoch in range(EPOCHS):
    model.train()

    for i, (mat, tgt, speed) in enumerate(train_data):
        mat = mat.to(DEVICE)
        tgt = tgt.to(DEVICE)
        speed = speed.to(DEVICE)
        optimizer.zero_grad()

        pre = model(mat, speed)
        loss_value = loss(pre, tgt)
        loss_value.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, steps, loss_value.item()))
