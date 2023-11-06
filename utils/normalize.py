import numpy as np
import torch

# 对于时间序列，在时间维度上做mean-variance normalization
# x的维度为(N, L, D), mask 的维度为(N, 1)
def ts_mean_var_norm(x, mask):
    mean = torch.nanmean(x, dim=1, keepdim=True)
    std = torch.std(x, dim=1, keepdim=True)
    mask = mask.reshape(-1, 1, 1)
    return (x - mean) / (std + 1e-8) * mask


def ts_min_max_norm(x, mask):
    min = torch.min(x, dim=1, keepdim=True)[0]
    max = torch.max(x, dim=1, keepdim=True)[0]
    mask = mask.reshape(-1, 1, 1)
    return 2 * ((x - min) / (max - min + 1e-8) - 0.5) * mask