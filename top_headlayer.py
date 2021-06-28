import torch
from torch import nn


class TopHeadLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, device, d_categories=24):  # d_categories=11):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward).to(device)
        self.linear2 = nn.Linear(d_feedforward, d_model).to(device)
        self.linear3 = nn.Linear(d_model, d_categories).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def forward(self, batch):
        # batch `(batch_size, d_model)`
        inner = torch.tanh(self.linear1(batch))
        wx = torch.tanh(self.linear2(inner))

        return self.softmax(self.linear3(wx))
