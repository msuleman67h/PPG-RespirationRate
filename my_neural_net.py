import torch.nn as nn
import torch


class MyNeuralNet(nn.Module):
    def __init__(self, sequence_length, output_size, input_size, hidden_size=200, num_layers=3):
        super(MyNeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1,))
        self.bn = nn.BatchNorm1d(num_features=1)
        self.max_pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1,))

        self.fc1 = nn.Linear(self.hidden_size * sequence_length, 8 * 1000)
        self.fc2 = nn.Linear(8 * 1000, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # cnn_out = self.bn(self.conv1(x))
        # cnn_out = self.leaky_relu(cnn_out)
        # cnn_out = self.max_pool(cnn_out)
        #
        # cnn_out = self.bn(self.conv2(cnn_out))
        # cnn_out = self.leaky_relu(cnn_out)
        # cnn_out = self.max_pool(cnn_out)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        lstm_out, _ = self.lstm1(x, (h0, c0))

        combined = self.fc1(lstm_out.view(-1, self.hidden_size * self.sequence_length))
        combined = torch.relu(combined)
        combined = self.fc2(combined)
        combined = torch.relu(combined)
        return combined
