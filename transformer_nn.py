import torch
import torch.nn as nn


class MyTransformer(nn.Module):
    def __init__(self, encoder_layers, n_heads):
        super(MyTransformer, self).__init__()
        self.encoder_layers = encoder_layers
        self.n_heads = n_heads

        self.trans = nn.Transformer(nhead=n_heads, num_encoder_layers=encoder_layers, batch_first=True)

    def forward(self, src, target):
        return self.trans(src, target)
