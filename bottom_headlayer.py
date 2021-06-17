import torch
from torch import nn


class BottomHeadLayer(nn.Module):
    def __init__(self, opcode_size, operand_size, d_model, device):
        super().__init__()
        assert d_model % 3 == 0, "The dim of BottomEmbedding should be multiple of 3"

        self.per_op_dim = d_model // 3
        self.opcode_linear = nn.Linear(self.per_op_dim, opcode_size).to(device)
        self.operand_1_linear = nn.Linear(self.per_op_dim, operand_size).to(device)
        self.operand_2_linear = nn.Linear(self.per_op_dim, operand_size).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def forward(self, batch):
        # batch `(batch_size, max_length, d_model)`
        opcode_output = self.opcode_linear(batch[:, :, : self.per_op_dim])
        operand_1_output = self.operand_1_linear(
            batch[:, :, self.per_op_dim : self.per_op_dim * 2]
        )
        operand_2_output = self.operand_2_linear(batch[:, :, self.per_op_dim * 2 :])

        opcode_predict = self.softmax(opcode_output).unsqueeze(dim=-2)
        operand_1_predict = self.softmax(operand_1_output).unsqueeze(dim=-2)
        operand_2_predict = self.softmax(operand_2_output).unsqueeze(dim=-2)

        # `(batch_size, max_length, 3, vocab_size)`
        return torch.cat((opcode_predict, operand_1_predict, operand_2_predict), dim=-2)

