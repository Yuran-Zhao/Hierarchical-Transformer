import torch
from torch import nn


class BottomEmbedding(nn.Module):
    def __init__(
        self, opcode_size, operand_size, d_model, padding_idx, max_length, device
    ):
        super().__init__()
        assert d_model % 3 == 0, "The dim of BottomEmbedding should be multiple of 3"

        per_op_dim = d_model // 3
        self.opcode_embedding_layer = nn.Embedding(opcode_size, per_op_dim).to(device)
        self.operand1_embedding_layer = nn.Embedding(operand_size, per_op_dim).to(
            device
        )
        self.operand2_embedding_layer = nn.Embedding(operand_size, per_op_dim).to(
            device
        )
        self.position_embedding = nn.Embedding(max_length, d_model).to(device)
        self.position = torch.tensor(
            [i for i in range(max_length)], dtype=torch.long, device=device
        )

    def forward(self, batch):
        # batch_size, max_length, inst_size = batch.size()
        opcode_embs = self.opcode_embedding_layer(batch[:, :, 0])
        operand_1_embs = self.operand1_embedding_layer(batch[:, :, 1])
        operand_2_embs = self.operand2_embedding_layer(batch[:, :, 2])
        input_embs = torch.cat((opcode_embs, operand_1_embs, operand_2_embs), dim=-1)
        position = torch.clone(self.position).expand(batch.shape[0], -1)[
            :, : batch.shape[1]
        ]
        position_embs = self.position_embedding(position)
        return position_embs + input_embs

