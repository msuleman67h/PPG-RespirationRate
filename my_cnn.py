import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, in_channels=400):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=800, kernel_size=(5,))
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=800, kernel_size=(4,))
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=800, kernel_size=(3,))

        self.bn = nn.BatchNorm1d(num_features=800)
        self.max_pool = nn.MaxPool1d(4)

        self.fcl1 = nn.Linear(3200, 1500)
        self.fcl2 = nn.Linear(1500, 500)
        self.fcl3 = nn.Linear(500, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn(c1)
        c1 = torch.relu(c1)
        c1 = self.max_pool(c1)

        c2 = self.conv2(x)
        c2 = self.bn(c2)
        c2 = torch.relu(c2)
        c2 = self.max_pool(c2)

        c3 = self.conv3(x)
        c3 = self.bn(c3)
        c3 = torch.relu(c3)
        c3 = self.max_pool(c3)

        c1 = torch.flatten(c1)
        c2 = torch.flatten(c2)
        c3 = torch.flatten(c3)

        combined = torch.cat([c1, c2, c3], dim=0)

        combined = combined.view(-1, 3200)
        combined = self.fcl1(combined)
        combined = torch.relu(combined)

        combined = self.fcl2(combined)
        combined = torch.relu(combined)

        combined = self.fcl3(combined)
        combined = torch.relu(combined)
        return combined
