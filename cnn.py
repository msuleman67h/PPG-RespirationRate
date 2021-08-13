import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=(250, ), stride=(4, ))
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.max_pool = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(125, ))
        self.bn2 = nn.BatchNorm1d(num_features=128)

        self.fcl1 = nn.Linear(128, 30)
        self.fcl2 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = torch.sigmoid(x)
        x = self.max_pool(x)

        x = self.bn2(self.conv2(x))
        x = torch.sigmoid(x)
        x = self.max_pool(x)

        x = x.view(-1, 128)
        x = torch.sigmoid(self.fcl1(x))
        x = torch.sigmoid(self.fcl2(x))
        return x.sum().unsqueeze(0).unsqueeze(0)



