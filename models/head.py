import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim, head_dropout, num_patch, flatten=True):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.flatten = flatten
        if self.flatten:
            self.linear = nn.Linear(num_patch*d_model, output_dim)
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: [(bs * nvars) x length x d_model]
        output: [(bs * nvars) x output_dim]
        """
        if self.flatten:
            x = self.flatten(x)
        else:
            x = x[:, -1, :]         # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        return y


class PretrainHead(nn.Module):
    def __init__(self, d_model, output_dim, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: [(bs * nvars) x length x d_model]
        output: [(bs * nvars) x length x output_dim]
        """
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        return y