import torch.nn as nn
import torch


class CNN2D(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 14, (3, 3))
        self.conv2 = nn.Conv2d(14, 20, (4, 4))

        self.bn1 = nn.BatchNorm2d(14)
        self.bn2 = nn.BatchNorm2d(20)

        self.max_pool = nn.MaxPool2d(3, 3)

        self.fcl1 = nn.Linear(20 * 5 * 5, 200)
        self.fcl2 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.max_pool(x)

        x = x.view(-1, 20 * 5 * 5)
        combined = self.fcl1(x)
        combined = torch.relu(combined)

        combined = self.fcl2(combined)
        combined = torch.relu(combined)
        return combined
