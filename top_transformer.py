import logging
import math
import os
import pdb
import random

import numpy as np
import torch
from top_headlayer import TopHeadLayer
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TopTransformer(nn.Module):
    def __init__(
        self, d_model=96, n_head=8, num_layers=6, max_length=50, device="cuda:0"
    ):
        super().__init__()
        transformer_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model
        ).to(device)
        self.transformer = TransformerEncoder(transformer_layer, num_layers).to(device)
        self.inserted_vector = torch.randn(d_model, requires_grad=True, device=device)
        self.top_headlayer = TopHeadLayer(d_model, device=device)

    def forward(self, batch, mask):
        """batch `(batch_size, binary_max_length, d_model)`
            mask `(batch_size, binary_max_length)`
        """
        batch_size, binary_max_length, _ = mask.shape
        # pdb.set_trace()
        inserted = torch.clone(self.inserted_vector)
        inserted = inserted.expand(batch_size, 1, -1)
        ones = torch.ones(batch_size)
        mask = torch.cat((ones, mask), dim=1)
        batch = torch.cat((inserted, batch), dim=1)

        batch = batch.permute(1, 0, 2).contiguous()
        tmp = self.transformer(batch, src_key_padding_mask=mask)
        return self.top_headlayer(tmp[:, 0, :])
