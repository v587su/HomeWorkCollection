import torch
import torch.cuda
import torch.utils.data as data
import torch.nn as nn
import torch.optim
import torchvision
import torch.autograd
from torch.utils.data import random_split
import os

EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 64

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.layers_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.layers_fc = nn.Sequential(
            nn.Linear(in_features=3136, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=10),
            nn.ReLU(),
        )
        nn.init.kaiming_normal_(self.layers_conv[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layers_conv[2].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layers_conv[6].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layers_conv[8].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layers_fc[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layers_fc[3].weight, nonlinearity='relu')

    def forward(self, input):
        input.cuda(device)
        output_conv = self.layers_conv(input)
        output_conv = output_conv.view(output_conv.size(0), -1)
        output_fc = self.layers_fc(output_conv)
        return output_fc


file_path = os.path.dirname(os.path.abspath(__file__))

data_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
train_set = torchvision.datasets.MNIST(os.path.join(file_path, 'data'), download=True, train=True,
                                       transform=data_transform)
test_set = torchvision.datasets.MNIST(os.path.join(file_path, 'data'), download=True, train=False,
                                      transform=data_transform)
full_size = len(test_set)
test_set, val_set = random_split(test_set, [round(0.8 * full_size), round(0.2 * full_size)])

loader_train = data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
loader_val = data.DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE)
loader_test = data.DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)

model = LeNet()
model.cuda(device)
loss = nn.CrossEntropyLoss()
loss.cuda(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

steps = len(loader_train)

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

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, steps, loss_value.item()))

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

if final_acc > 0.985:
    torch.save(model, 'model.pkl')
else:
    print('The performance is not well enough!')
