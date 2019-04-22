import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import os

from torch.utils.data import random_split


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pre, real):
        pass


class BasicConv(nn.Module):
    def __init__(self, in_channels, alpha=1, layer_jump=False):
        super(BasicConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64 * alpha,
                      kernel_size=1, bias=True),
            nn.BatchNorm2d(64 * alpha),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * alpha, out_channels=64 * alpha,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64 * alpha),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * alpha, out_channels=256 * alpha,
                      kernel_size=1, bias=True),
            nn.BatchNorm2d(64 * alpha),
        )
        self.jump = nn.Sequential(
            nn.Conv2d(in_channels=64 * alpha, out_channels=256 * alpha,
                      kernel_size=1, bias=True),
            nn.BatchNorm2d(64 * alpha),
        ) if layer_jump else nn.Sequential()

    def forward(self, x):
        y = self.model(x)
        x_jump = self.jump(x)
        output = nn.ReLU(x_jump + y)
        return output


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                          stride=2, padding=3)
        pool1 = nn.MaxPool2d(3, 2)
        conv2 = [
            BasicConv(in_channels=64, alpha=1),
            BasicConv(in_channels=256, alpha=1),
            BasicConv(in_channels=256, alpha=1, layer_jump=True)
        ]
        conv3 = [
            BasicConv(in_channels=256, alpha=2),
            BasicConv(in_channels=512, alpha=2),
            BasicConv(in_channels=512, alpha=2),
            BasicConv(in_channels=512, alpha=2, layer_jump=True)
        ]
        conv4 = [
            BasicConv(in_channels=512, alpha=3),
            BasicConv(in_channels=1024, alpha=3),
            BasicConv(in_channels=1024, alpha=3),
            BasicConv(in_channels=1024, alpha=3),
            BasicConv(in_channels=1024, alpha=3),
            BasicConv(in_channels=1024, alpha=3, layer_jump=True)
        ]
        conv5 = [
            BasicConv(in_channels=1024, alpha=3),
            BasicConv(in_channels=2048, alpha=3),
            BasicConv(in_channels=2048, alpha=3, layer_jump=True)
        ]
        pool2 = nn.AvgPool2d(8)

        self.model_conv = nn.Sequential(
            conv1,
            pool1,
            *conv2,
            *conv3,
            *conv4,
            *conv5,
            pool2,
        )

        self.model_linear = nn.Sequential(
            nn.Linear(2048, 10),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.model_conv(x)
        out = out.view(out.size(0), -1)
        out = self.model_linear(out)
        return out


file_path = os.path.dirname(os.path.abspath(__file__))

data_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
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

print(loader_train)