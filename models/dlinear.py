import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ffn import MLP_linear


class TLinear(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, args):
        super(TLinear, self).__init__()
        self.args = args
        self.seq_len = self.args.window_length
        self.pred_len = self.args.pred_length
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = self.args.char_dim
        self.temporal_linear = nn.Linear(self.seq_len, self.pred_len)
        self.char_linear = nn.Linear(self.args.char_dim, 1)
        # self.temporal_linear = MLP_linear(args=None, out_dim=1, d_model=64, input_dim=self.seq_len, num_enc_layers=2, dropout=0.1)
        # self.char_linear = MLP_linear(self.args)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        N, T, C = x.shape
        x = self.temporal_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.char_linear(x)
        # x = self.temporal_linear(x.permute(0, 2, 1).reshape(N * C, T)).reshape(N, C, 1).permute(0, 2, 1)
        # x = self.char_linear(x.reshape(N, C)).reshape(N, 1, 1)
        return x  # [Batch, Output length, Channel]