import torch
from torch import nn


class MiddleHeadLayer(nn.Module):
    def __init__(self, d_model, device):
        super().__init__()
        self.linear = nn.Linear(d_model, 2).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def forward(self, batch):
        # batch `(batch_size, d_model)`
        output = self.linear(batch)

        # `(batch_size, 2)`
        return self.softmax(output)

