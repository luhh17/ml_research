import numpy as np
import torch
import torch_package.functions as functions
#import math
EPS = torch.finfo().eps
ANN_CORR_WEIGHT = 0.9
#=================================================================================#
#=============================== Loss Functions =================================#
#=================================================================================#


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


class RetMSELoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, weight_mat, return_mat, mask_mat):
        weight_mat = mask_mat * weight_mat
        rp = torch.sum(weight_mat * return_mat, dim=1)  
        rp_mean = torch.mean(rp)
        rp_std = torch.std(rp)
        annual_sr = np.sqrt(250) * rp_mean / rp_std
        loss = torch.sum((weight_mat - return_mat)**2) / torch.sum(mask_mat)
        return loss, rp_mean, rp_std, annual_sr

class ANNLoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
    def forward(self, weight_mat, return_mat, mask_mat):
        weight_mat = mask_mat * weight_mat
        denominator = torch.sum(mask_mat)
        
        # Basic Attributes
        rp = torch.sum(weight_mat * return_mat, dim=1)  
        rp_mean = torch.mean(rp)
        rp_std = torch.std(rp)
        annual_sr = np.sqrt(250) * rp_mean / rp_std
        
        # MSE Loss
        mse_loss = torch.sum((weight_mat - return_mat)**2) / torch.sum(mask_mat)
        
        # Correlation Loss
        y_true = return_mat - torch.sum(return_mat, dim=(0, 1), keepdim=True) / denominator
        y_pred = weight_mat - torch.sum(weight_mat, dim=(0, 1), keepdim=True) / denominator
        cov = torch.sum(y_true * y_pred)
        std_prod = torch.sqrt(torch.sum(y_true**2) * torch.sum(y_pred**2))
        corr_loss = - cov / (std_prod + EPS)
        
        # Loss
        loss = ANN_CORR_WEIGHT * corr_loss + (1-ANN_CORR_WEIGHT) * mse_loss
        
        return loss, rp_mean, rp_std, annual_sr
    
class UtilityLoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, weight_mat, return_mat, mask_mat):
        weight_mat = mask_mat * weight_mat
        rp = torch.sum(weight_mat * return_mat, dim=1)  
        rp_mean = torch.mean(rp)
        rp_std = torch.std(rp)
        annual_sr = np.sqrt(250) * rp_mean / rp_std
        loss =  - rp_mean + 10 * rp_std**2
        return loss, rp_mean, rp_std, annual_sr

def get_regular_loss(model, how, rate):
    regu_loss = 0
    if how == 'l1':
        for param in model.parameters():
            regu_loss += torch.sum(torch.abs(param))

    if how == 'l2':
        for param in model.parameters():
            regu_loss += torch.sum(torch.square(param))       

    if how == 'l1_l2':
        for param in model.parameters():
            regu_loss += torch.sum(torch.abs(param)) + torch.sum(torch.square(param)) 
    
    return regu_loss * rate

