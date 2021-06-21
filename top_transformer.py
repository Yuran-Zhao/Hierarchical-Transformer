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
        middle_transformer,
        d_model=96,
        n_head=8,
        num_layers=6,
        max_length=50,
        device="cuda:0",
    ):
        super().__init__()
        self.middle_transformer = middle_transformer
        transformer_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model
        ).to(device)
        self.transformer = TransformerEncoder(transformer_layer, num_layers).to(device)
        self.inserted_vector = torch.randn(d_model, requires_grad=True, device=device)
        self.top_headlayer = TopHeadLayer(d_model, device=device)

    def forward(self, input_ids, mask_for_bottom, mask_for_middle, mask_for_binary):
        """input_ids `(batch_size, binary_max_size, function_max_size, block_max_size, inst_max_size)`
            mask_for_bottom `(batch_size, binary_max_size, function_max_size, block_max_size)`
            mask_for_middle `(batch_size, binary_max_size, function_max_size)`
            mask_for_binary `(batch_size, binary_max_size)`
        """
        # pdb.set_trace()
        (
            batch_size,
            binary_max_size,
            function_max_size,
            block_max_size,
            _,
        ) = input_ids.shape
        input_ids = torch.reshape(
            input_ids, (-1, function_max_size, block_max_size, inst_max_size)
        ).contiguous()
        mask_for_bottom = torch.reshape(
            mask_for_bottom, (-1, function_max_size, block_max_size)
        ).contiguous()
        mask_for_middle = torch.reshape(
            mask_for_middle, (-1, function_max_size)
        ).contiguous()

        # middle_output `(batch_size * binary_max_size, d_model)`
        middle_output = self.middle_transformer(
            input_ids, mask_for_bottom, mask_for_middle
        )
        middle_output = torch.reshape(
            middle_output, (batch_size, binary_max_size, d_model)
        ).contiguous()

        inserted = torch.clone(self.inserted_vector)
        inserted = inserted.expand(batch_size, 1, -1)
        ones = torch.ones(batch_size).unsqueeze(dim=1).to(mask_for_binary.device)
        mask_for_binary = torch.cat((ones, mask_for_binary), dim=1)
        batch = torch.cat((inserted, middle_output), dim=1)

        batch = batch.permute(1, 0, 2).contiguous()
        # tmp `(binary_max_size + 1, batch_size, d_model)`
        tmp = self.transformer(batch, src_key_padding_mask=mask_for_binary)
        tmp = tmp.permute(1, 0, 2).contiguous()

        return self.top_headlayer(tmp[:, 0, :])
