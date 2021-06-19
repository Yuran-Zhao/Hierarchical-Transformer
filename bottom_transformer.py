import logging
import math
import os
import pdb
import random

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from bottom_embedding import BottomEmbedding
from bottom_headlayer import BottomHeadLayer


class BottomTransformer(nn.Module):
    def __init__(
        self,
        opcode_size,
        operand_size,
        padding_idx,
        d_model=96,
        n_head=8,
        num_layers=6,
        max_length=251,
        device="cuda:0",
    ):
        super().__init__()
        self.bottom_embedding = BottomEmbedding(
            opcode_size, operand_size, d_model, padding_idx, max_length, device
        )
        transformer_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model
        ).to(device)
        self.transformer = TransformerEncoder(transformer_layer, num_layers).to(device)
        self.bottom_headlayer = BottomHeadLayer(
            opcode_size, operand_size, d_model, device
        )

    def forward(self, batch, masks, header=False):
        # pdb.set_trace()
        # batch `(batch_size, max_length, inst_size)`
        input_embs = self.bottom_embedding(batch)

        # in the input to the transformer 'batch_size' should be in the second dimension
        input_embs = input_embs.permute(1, 0, 2).contiguous()

        tmp = self.transformer(input_embs, src_key_padding_mask=masks.bool())

        # re-permute to make the 'batch_size' in the first dimension
        tmp = tmp.permute(1, 0, 2).contiguous()

        if header:
            return self.bottom_headlayer(tmp)

        return tmp[:, 0, :]
