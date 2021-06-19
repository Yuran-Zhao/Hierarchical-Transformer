import torch
from torch import nn


class MiddleHeadLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, device):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward).to(device)
        self.linear2 = nn.Linear(d_feedforward, d_model).to(device)
        # self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def forward(self, batch):
        # batch `(batch_size, d_model)`
        inner = torch.tanh(self.linear1(batch))
        wx = torch.tanh(self.linear2(inner))

        return torch.sigmoid(torch.diag(torch.mm(wx, x), diagonal=0))

