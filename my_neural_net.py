import torch
import torch.nn as nn


class MyNeuralNet(nn.Module):
    def __init__(self, sequence_length, output_size, input_size, hidden_size, num_layers):
        super(MyNeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=10000, num_heads=10000, batch_first=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.view(1, 250, -1)
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        lstm_out, _ = self.lstm1(x, (h0, c0))
        lstm_out = lstm_out.flatten().unsqueeze(0)

        out, weight = self.attn(query=lstm_out, key=lstm_out, value=lstm_out)

        combined = torch.cat([lstm_out], dim=1)
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        combined = self.fc2(combined)
        combined = torch.sigmoid(combined)
        return combined
