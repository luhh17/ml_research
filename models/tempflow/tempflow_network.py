from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import sys
from models.tempflow.flows import RealNVP, MAF


class TempFlowNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rnn = nn.LSTM(
            input_size=742,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )

        if self.args.flow == 'realnvp':
            self.flow = RealNVP(
                input_size=742,
                n_blocks=5,
                n_hidden=3,
                hidden_size=100,
                cond_label_size=256,
            )
        elif self.args.flow == 'maf':
            self.flow = MAF(
                input_size=742,
                n_blocks=5,
                n_hidden=3,
                hidden_size=100,
                cond_label_size=256,
            )


    def forward(self, past_time_series, target, past_covariates=None, begin_state=None, mask=None):
        # N,L,D∗H, D∗num_layers,N,H
        past_time_series = past_time_series.permute(0, 2, 1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)
        outputs, state = self.rnn(past_time_series)
        likelihoods = -self.flow.log_prob(target, outputs[:, -1, :], mask).unsqueeze(-1)
        loss = torch.mean(likelihoods)
        mse_loss = []
        pred_list = []
        for i in range(100):
            pred = self.flow.sample(target.shape, outputs[:, -1, :   ]) * mask
            mse = torch.mean(((pred - target) ** 2) * mask)
            mse_loss.append(mse)
            pred_list.append(pred)
        avg_pred = torch.mean(torch.stack(pred_list), dim=0)
        mse = torch.mean(((avg_pred - target) ** 2) * mask)
        return loss, mse, pred_list

    def infer(self, past_time_series, past_covariates=None, begin_state=None):
        # N,L,D∗H, D∗num_layers,N,H
        outputs, state = self.rnn(past_time_series, begin_state=begin_state)
        pred = self.flow.sample(target.shape, outputs)
        return pred
