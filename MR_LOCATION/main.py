import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os.path as path
from util import *

seed = 235
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


crt_path = path.dirname(path.abspath(__file__))
data_path = path.join(crt_path, 'data_file', 'train_2g.csv')
trail_path = path.join(crt_path, 'data_file', 'trail.csv')
matrix_path = path.join(crt_path, 'data_file', 'matrix.csv')

data = pd.read_csv(data_path)
trail = np.loadtxt(trail_path, delimiter=',')
matrix = pd.read_csv(matrix_path, header=None)
# train_idx, val_idx = train_test_split(trail, test_size=0.2, shuffle=True)
train_idx = trail
train_data = DataLoader(Trails(data, train_idx, matrix),
                        batch_size=BATCH_SIZE)
# val_data = DataLoader(Trails(data, val_idx, matrix), batch_size=BATCH_SIZE)

model = CLSTM().double()
loss = nn.MSELoss()
if USE_CUDA:
    model.cuda()
    loss.cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_NORM)

steps = len(train_data)
total_loss = []
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
        total_loss.append(loss_value.item())
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, steps, loss_value.item()))
#
# model.eval()
# losses = []
# for i, (mat, tgt, speed) in enumerate(val_data):
#     mat = mat.to(device=DEVICE).double()
#     tgt = tgt.to(device=DEVICE).double()
#     speed = speed.to(device=DEVICE).double()
#     pre = model(mat, speed)
#     loss_value = loss(pre, tgt)
#     losses.append(loss_value.item())
#     pres = pre.cpu().detach().numpy().reshape(-1, 2)
#     tgts = tgt.cpu().detach().numpy().reshape(-1, 2)
torch.save(model, 'model.pkl')
# pd.DataFrame(np.column_stack((pres, tgts))).to_csv('result.csv', header=False, index=False)
pd.DataFrame(total_loss).to_csv('total_loss.csv', header=False, index=False)
# print('mean loss:', np.mean(np.array(losses)))
# print('max loss:', np.max(np.array(losses)))
