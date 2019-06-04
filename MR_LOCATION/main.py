import torch
import torch.nn as nn
import torch.optim
import pandas
import numpy as np

TRAIL_LENGTH_MAX = 30


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

    def forward(self, x):
        pass


loss = nn.MSELoss()
model =
optim = torch.optim.adam()
