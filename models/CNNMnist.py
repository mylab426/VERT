import time

from torch import nn
import torch.nn.functional as F


class MLPmnist(nn.Module):
    def __init__(self):
        super(MLPmnist, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 10)
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x, F.log_softmax(x, dim=1)


class cnnmnist(nn.Module):
    def __init__(self):
        super(cnnmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        # 数据摊平
        x2 = x2.view(-1, x2.shape[1] * x2.shape[2] * x2.shape[3])
        x3 = F.relu(self.fc1(x2))
        x3 = F.dropout(x3, training=self.training)
        x4 = self.fc(x3)
        return x4, F.log_softmax(x4, dim=1)


class SimpleMnist(nn.Module):
    def __init__(self):
        super(SimpleMnist, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return [], F.log_softmax(x, dim=1)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x2 = x2.view(-1, 16 * 5 * 5)
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = self.fc3(x4)
        return x5, F.log_softmax(x5, dim=1)


class LeNet_plus(nn.Module):
    def __init__(self):
        super(LeNet_plus, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x =self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        x = self.fc(x)
        return 1, x
