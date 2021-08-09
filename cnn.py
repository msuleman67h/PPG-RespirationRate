import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, in_channels=1250, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=1250, kernel_size=(1, ))
        self.bn1 = nn.BatchNorm1d(num_features=1250)
        self.conv2 = nn.Conv1d(in_channels=1250, out_channels=1250, kernel_size=(1, ))

    def forward(self, x):
        x = torch.sigmoid(self.bn1(self.conv1(x)))
        x = self.bn1(x)
        x = torch.sigmoid(self.conv2(x))
        return x


