import torch
import torch.nn as nn


class MyNeuralNet(nn.Module):
    def __init__(self, sequence_length, output_size, input_size, hidden_size, num_layers):
        super(MyNeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.conv1 = nn.Conv1d(in_channels=250, out_channels=50, kernel_size=(5,))
        self.bn = nn.BatchNorm1d(num_features=50)
        self.mxp = nn.MaxPool1d(3)

        self.conv2 = nn.Conv1d(in_channels=150, out_channels=50, kernel_size=(3,))

        self.conv3 = nn.Conv1d(in_channels=100, out_channels=50, kernel_size=(5,))

        self.conv4 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=(3,))

        self.conv5 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=(5,))

        self.fc1 = nn.Linear(14700, 10000)
        self.fc2 = nn.Linear(10000, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        cnn_out = [self.bn(self.conv1(x.view(1, 250, -1)))]
        cnn_out[0] = torch.sigmoid(cnn_out[0])
        cnn_out[0] = self.mxp(cnn_out[0])

        cnn_out.append(self.bn(self.conv2(x.view(1, 150, -1))))
        cnn_out[1] = torch.sigmoid(cnn_out[1])
        cnn_out[1] = self.mxp(cnn_out[1])

        cnn_out.append(self.bn(self.conv3(x.view(1, 100, -1))))
        cnn_out[2] = torch.sigmoid(cnn_out[2])
        cnn_out[2] = self.mxp(cnn_out[2])

        cnn_out.append(self.bn(self.conv4(x.view(1, 50, -1))))
        cnn_out[3] = torch.sigmoid(cnn_out[3])
        cnn_out[3] = self.mxp(cnn_out[3])

        cnn_out.append(self.bn(self.conv5(x.view(1, 25, -1))))
        cnn_out[4] = torch.sigmoid(cnn_out[4])
        cnn_out[4] = self.mxp(cnn_out[4])

        cnn_out = [torch.flatten(c_o) for c_o in cnn_out]

        x = x.view(1, 125, -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        lstm_out, _ = self.lstm1(x, (h0, c0))
        lstm_out = lstm_out.flatten()

        combined = torch.cat([lstm_out, cnn_out[0], cnn_out[1], cnn_out[2], cnn_out[3], cnn_out[4]])
        combined = self.fc1(combined)
        combined = torch.sigmoid(combined)
        combined = self.fc2(combined)
        combined = torch.sigmoid(combined)
        return combined
