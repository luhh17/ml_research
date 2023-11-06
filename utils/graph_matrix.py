import torch
import numpy as np
import torch.nn as nn


def norm_adj(adj, add_eye=True):
    # eye = torch.eye(adj.shape[1]).repeat(adj.shape[0], 1, 1)
    if add_eye:
        eye = torch.eye(adj.shape[1]).repeat(adj.shape[0], 1, 1).to(adj.device)
        adj = normalize(adj + eye)
    else:
        adj = normalize(adj)
    return adj


def dummy_adj(adj, threshold=0.1, add_eye=True):
    # treat edegs with weight < threshold as 0 weight and 1 otherwise
    adj[adj < threshold] = 0
    adj[adj >= threshold] = 1
    if add_eye:
        eye = torch.eye(adj.shape[1]).repeat(adj.shape[0], 1, 1).to(adj.device)
        adj = normalize(adj + eye)
    else:
        adj = normalize(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    # input B x N x N
    rowsum = torch.sum(mx, dim=2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = []
    for b in range(mx.shape[0]):
        _r_mat_inv = torch.diag(r_inv[b])
        r_mat_inv.append(_r_mat_inv)
    r_mat_inv = torch.stack(r_mat_inv, dim=0)
    # print(r_mat_inv.dtype, mx.dtype)
    mx = torch.bmm(r_mat_inv, mx)
    return mx

