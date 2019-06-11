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
    def __init__(self, d, t, m, normalize=None):
        super(Trails, self).__init__()
        label_list = ['Longitude', 'Latitude']

        dd = d.copy()
        mm = m.copy()

        # 对数据进行归一化处理
        dd['Longitude'] = dd['Longitude'].sub(MIN_LON).div(MAX_LON - MIN_LON)
        dd['Latitude'] = dd['Latitude'].sub(MIN_LAT).div(MAX_LAT - MIN_LAT)
        mm = mm.values.reshape(-1, 5)
        for ii in range(5):
            x = mm[:, ii]
            mm[:, ii] = (x - np.mean(x)) / np.std(x)
        mm = mm.reshape(-1, 6, 5)
        self.mat = mm
        self.tra = t
        self.tgt = dd[label_list].values
        self.speed = dd['Speed'].values

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
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5,
                      padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,
                      padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=3,
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),
        )
        self.linear = nn.Linear(12, 2)
        self.relu = nn.LeakyReLU()

        normalize_layer = [0, 2, 6, 8]
        for i in normalize_layer:
            nn.init.kaiming_normal_(self.model[i].weight,
                                    nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear.weight,
                                nonlinearity='leaky_relu')

    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), output.size(1), -1)
        output = self.linear(output)
        output = self.relu(output)
        return output


class CLSTM(nn.Module):
    def __init__(self):
        super(CLSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(input_size=3, hidden_size=2,
                            batch_first=True)

    def forward(self, m, speed):
        # 先过cnn，出来的结果concat speed然后喂进lstm
        # m = m.permute(1, 0, 2, 3)
        # fea_list = []
        # for batch in m:
        #     fea = self.cnn(batch.view(-1, 1, 6, 5))
        #     fea_list.append(fea.view(-1, 1, 2))
        # fea_ts = torch.cat(fea_list, dim=1)

        fea = self.cnn(m)
        speed_size = speed.size()
        fea_speed = torch.cat((fea, speed.view(speed_size[0], -1, 1)),
                              dim=2)
        result, _ = self.lstm(fea_speed)
        return result


crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
trail_path = path.join(crt_path, 'data_file', 'trail.csv')
matrix_path = path.join(crt_path, 'data_file', 'matrix.csv')

data = pd.read_csv(data_path)
trail = np.loadtxt(trail_path, delimiter=',')
matrix = pd.read_csv(matrix_path, header=None)
train_idx, val_idx = train_test_split(trail, test_size=0.2, shuffle=True)

train_data = DataLoader(Trails(data, train_idx, matrix),
                        batch_size=BATCH_SIZE)
val_data = DataLoader(Trails(data, val_idx, matrix), batch_size=BATCH_SIZE)

model = CLSTM().double()
loss = nn.MSELoss()
if USE_CUDA:
    model.cuda()
    loss.cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

steps = len(train_data)
for epoch in range(EPOCHS):
    model.train()

    for i, (mat, tgt, speed) in enumerate(train_data):
        mat = mat.to(device=DEVICE).double()
        tgt = tgt.to(device=DEVICE).double()
        speed = speed.to(device=DEVICE).double()
        optimizer.zero_grad()

        pre = model(mat, speed)
        loss_value = loss(pre, tgt)
        loss_value.backward()
        optimizer.step()
        if i % 100 == 2:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, steps, loss_value.item()))
