import logging
import math
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from middle_headlayer import MiddleHeadLayer


class MiddleTransformer(nn.Module):
    def __init__(
        self, d_model=96, n_head=8, num_layers=6, max_length=50,
    ):
        super().__init__()
        transformer_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model
        )
        self.transformer = TransformerEncoder(transformer_layer, num_layers)
        self.middle_headlayer = MiddleHeadLayer(d_model)

    def forward(self, batch, mask):
        # batch `(batch_size, max_length, d_model)`
        tmp = self.transformer(batch, src_key_padding_mask=mask)
        return self.middle_headlayer(tmp[:, 0, :])
