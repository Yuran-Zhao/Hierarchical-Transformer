import logging
import math
import os
import pdb
import random

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from top_headlayer import TopHeadLayer


class TopTransformer(nn.Module):
    def __init__(
        self,
        # middle_transformer,
        d_model=1280,
        n_head=8,
        num_layers=6,
        # max_length=50,
        device="cuda:0",
    ):
        super().__init__()
        self.d_model = d_model
        # self.middle_transformer = middle_transformer
        transformer_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model
        ).to(device)
        self.transformer = TransformerEncoder(transformer_layer, num_layers).to(device)
        self.inserted_vector = torch.randn(d_model, requires_grad=True, device=device)
        self.top_headlayer = TopHeadLayer(d_model, 4 * d_model, device=device)

    def forward(self, input_ids):
        """input_ids `(batch_size, binary_max_size, d_model)`
        """
        # pdb.set_trace()
        batch_size, binary_max_size, d_model = input_ids.shape

        inserted = torch.clone(self.inserted_vector)
        inserted = inserted.expand(batch_size, 1, -1)

        batch = torch.cat((inserted, input_ids), dim=1)

        batch = batch.permute(1, 0, 2).contiguous()
        # tmp `(binary_max_size + 1, batch_size, d_model)`
        tmp = self.transformer(batch)
        tmp = tmp.permute(1, 0, 2).contiguous()

        return self.top_headlayer(tmp[:, 0, :])
