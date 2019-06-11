import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.optim
import os
import argparse
import numpy as np

from torch.utils.data import random_split
from torchvision import transforms

EPOCHS = 30
LEARNING_RATE = 0.01
BATCH_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--loss', type=int, default=0)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.SmoothL1Loss()

    def forward(self, pre, real):
        labels_onehot = torch.FloatTensor(real.shape[0], 10).to(device)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)
        my_loss = self.ce(pre, real) + self.l1(pre, labels_onehot)
        return my_loss


class AverageLoss(nn.Module):
    def __init__(self):
        super(AverageLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pre, real):
        labels_onehot = torch.FloatTensor(real.shape[0], 10).to(device)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)
        labels_onehot = 0.9 * labels_onehot + 0.1 * (1 - labels_onehot)
        print(self.kl(pre, labels_onehot))
        print(torch.mean(torch.log(pre)))
        print(torch.log(pre))
        average_loss = self.ce(pre, real) + self.mse(pre, labels_onehot)
        return average_loss / 2


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pre, real):
        labels_onehot = torch.FloatTensor(real.shape[0], 10).to(device)
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)
        labels_onehot = 0.9 * labels_onehot + 0.1 * (1 - labels_onehot)
        loss = self.mse(pre, labels_onehot)
        return


class BasicConv(nn.Module):
    def __init__(self, in_channels, alpha=1, layer_jump=False):
        super(BasicConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16 * alpha,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(16 * alpha),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * alpha, out_channels=16 * alpha,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16 * alpha),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * alpha, out_channels=64 * alpha,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(64 * alpha),
        )
        self.jump = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64 * alpha,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(64 * alpha),
        ) if layer_jump else nn.Sequential()
        self.relu = nn.ReLU()

        nn.init.kaiming_normal_(self.model[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.model[3].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.model[6].weight, nonlinearity='relu')
        if layer_jump:
            nn.init.kaiming_normal_(self.jump[0].weight, nonlinearity='relu')

    def forward(self, x):
        y = self.model(x)
        x_jump = self.jump(x)
        out = self.relu(x_jump + y)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                          stride=1, padding=1)
        pool1 = nn.MaxPool2d(3, 2)
        conv2 = [
            BasicConv(in_channels=16, alpha=1, layer_jump=True),
            BasicConv(in_channels=64, alpha=1),
            BasicConv(in_channels=64, alpha=1, layer_jump=True)
        ]
        conv3 = [
            BasicConv(in_channels=64, alpha=2, layer_jump=True),
            BasicConv(in_channels=128, alpha=2),
            BasicConv(in_channels=128, alpha=2),
            BasicConv(in_channels=128, alpha=2, layer_jump=True)
        ]
        conv4 = [
            BasicConv(in_channels=128, alpha=4, layer_jump=True),
            BasicConv(in_channels=256, alpha=4),
            BasicConv(in_channels=256, alpha=4),
            BasicConv(in_channels=256, alpha=4),
            BasicConv(in_channels=256, alpha=4),
            BasicConv(in_channels=256, alpha=4, layer_jump=True)
        ]
        conv5 = [
            BasicConv(in_channels=256, alpha=8, layer_jump=True),
            BasicConv(in_channels=512, alpha=8),
            BasicConv(in_channels=512, alpha=8, layer_jump=True)
        ]
        pool2 = nn.AvgPool2d(8)

        self.model_conv = nn.Sequential(
            conv1,
            pool1,
            nn.ReLU(),
            *conv2,
            *conv3,
            *conv4,
            *conv5,
            pool2,
        )

        self.model_linear = nn.Sequential(
            nn.Linear(512, 10),
        )

        nn.init.kaiming_normal_(self.model_conv[0].weight, nonlinearity='relu')

    def forward(self, input):
        input.cuda(device)
        out = self.model_conv(input)
        out = out.view(out.size(0), -1)
        out = self.model_linear(out)
        return out


file_path = os.path.dirname(os.path.abspath(__file__))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
])
train_set = torchvision.datasets.CIFAR10(os.path.join(file_path, 'data'),
                                         download=True, train=True,
                                         transform=data_transform)
test_set = torchvision.datasets.CIFAR10(os.path.join(file_path, 'data'),
                                        download=True, train=False,
                                        transform=data_transform)
full_size = len(test_set)
test_set, val_set = random_split(test_set, [round(0.8 * full_size),
                                            round(0.2 * full_size)])

loader_train = data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
loader_val = data.DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE)
loader_test = data.DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

model = ResNet()
model.cuda(device)
if args.loss == 0:
    loss = MyLoss()
elif args.loss == 1:
    loss = nn.CrossEntropyLoss()
elif args.loss == 2:
    loss = MSELoss()
elif args.loss == 3:
    loss = AverageLoss()

loss.cuda(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

steps = len(loader_train)
losses = []
acces = []

for epoch in range(EPOCHS):
    model.train()
    for i, (imgs, labels) in enumerate(loader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)
        loss_value = loss(output, labels)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, steps, loss_value.item()))
        losses.append(loss_value.item())

    model.eval()
    eval_acc = 0
    for i, (imgs, labels) in enumerate(loader_val):
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum().item()
        acc = num_correct / imgs.shape[0]
        eval_acc += acc
    print('Val ACC: {}'.format(eval_acc / len(loader_val)))
    acces.append((eval_acc / len(loader_val)))

model.eval()
eval_acc = 0
for i, (imgs, labels) in enumerate(loader_test):
    imgs = imgs.to(device)
    labels = labels.to(device)
    output = model(imgs)
    _, pred = output.max(1)
    num_correct = (pred == labels).sum().item()
    acc = num_correct / imgs.shape[0]
    eval_acc += acc

final_acc = eval_acc / len(loader_test)
print('Test ACC: {}'.format(final_acc))

torch.save(model, 'model_{}.pkl'.format(args.loss))
loss_np = np.array(losses, dtype=np.float)
acc_np = np.array(acces, dtype=np.float)
np.savetxt('loss_{}.csv'.format(args.loss), loss_np, delimiter=None)
np.savetxt('acc_{}.csv'.format(args.loss), acc_np, delimiter=None)
