import torch
from torch import nn


class MiddleHeadLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, batch):
        # batch `(batch_size,, d_model)`
        output = self.linear(batch)

        # `(batch_size, 2)`
        return self.softmax(output)

