import torch
import numpy as np


class RetMSELoss(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def forward(self, pred, ret, mask):
        pred = pred.reshape(-1, self.args.pred_length)
        ret = ret.reshape(-1, self.args.pred_length)
        mask = mask.reshape(-1, self.args.pred_length)
        pred = mask * pred
        loss = torch.sum((pred - ret) ** 2) / torch.sum(mask)
        return loss


class SRLoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, weight_mat, return_mat, mask_mat):
        weight_mat = mask_mat * weight_mat
        weight_mat = functions.torch_get_adjusted_weight(weight_mat)
        rp = torch.sum(weight_mat * return_mat, dim=1)
        rp_mean = torch.mean(rp)
        rp_std = torch.std(rp)
        annual_sr = np.sqrt(250) * rp_mean / rp_std
        loss = - rp_mean / rp_std
        return loss, rp_mean, rp_std, annual_sr

