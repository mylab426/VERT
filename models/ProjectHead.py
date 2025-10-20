
from torch import nn
import torch.nn.functional as F


class mnist_project_head(nn.Module):
    def __init__(self, input, output):
        super(mnist_project_head, self).__init__()
        self.fc1 = nn.Linear(input, output)

    def forward(self, x):
        x = self.fc1(x)
        return x


class mnist_predictor(nn.Module):
    def __init__(self, input, output):
        super(mnist_predictor, self).__init__()
        self.fc1 = nn.Linear(input, output)
        self.fc2 = nn.Linear(output, output)
        self.fc3 = nn.Linear(output, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        x = F.relu(self.fc2(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        # return F.softmax(x)
        return x