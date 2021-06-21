import logging
import math
import os
import pdb
import random

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# from middle_headlayer import MiddleHeadLayer


class MiddleTransformer(nn.Module):
    def __init__(
        self,
        bottom_transformer,
        d_model=96,
        n_head=8,
        num_layers=6,
        max_length=50,
        device="cuda:0",
    ):
        super().__init__()
        self.bottom_transformer = bottom_transformer
        self.device = device
        transformer_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model
        ).to(device)
        self.transformer = TransformerEncoder(transformer_layer, num_layers).to(device)
        self.inserted_vector = torch.randn(d_model, requires_grad=True, device=device)
        # self.middle_headlayer = MiddleHeadLayer(d_model, device=device)

    def forward(self, input_ids, mask_for_bottom, mask_for_function):
        """input_ids `(batch_size, function_max_size, block_max_size, 3)`
            mask_for_bottom `(batch_size, function_max_size, block_max_size)`
            mask_for_function `(batch_size, function_max_size)`
        """
        (batch_size, function_max_size, block_max_size, inst_size,) = input_ids.shape

        input_ids.to(self.device)
        mask_for_bottom.to(self.device)
        mask_for_function.to(self.device)
        input_ids = torch.reshape(
            input_ids, (-1, block_max_size, inst_size)
        ).contiguous()
        mask_for_bottom = torch.reshape(
            mask_for_bottom, (-1, block_max_size)
        ).contiguous()

        # bottom_output `(batch_size * function_max_size, d_model)`
        # pdb.set_trace()
        bottom_output = self.bottom_transformer(input_ids, None)  # mask_for_bottom)
        bottom_output = bottom_output.reshape(
            batch_size, function_max_size, -1
        ).contiguous()

        inserted = torch.clone(self.inserted_vector)
        inserted = inserted.expand(batch_size, 1, -1)
        ones = torch.ones(batch_size).unsqueeze(dim=1).to(mask_for_function.device)

        mask_for_function = torch.cat((ones, mask_for_function), dim=1)
        batch = torch.cat((inserted, bottom_output), dim=1)
        batch = batch.permute(1, 0, 2).contiguous()
        tmp = self.transformer(batch, None)  # src_key_padding_mask=mask_for_function.bool())
        tmp = tmp.permute(1, 0, 2).contiguous()

        return tmp[:, 0, :]
