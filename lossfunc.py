import numpy as np
import torch
import pdb
epsilon = torch.finfo().eps
ANN_CORR_WEIGHT = 0.9

def torch_get_adjusted_weight(Wt, thres=[0.8, 1.2]):
    # Threshold
    thres = torch.tensor(thres)
    # Get the Long and Short postitions seperately
    Wt_pos = torch.relu(Wt)
    Wt_neg = - torch.relu(-Wt)
    # Scale of long and short positions in the raw portfolio, [T,N,1]
    scl_pos = torch.sum(Wt_pos, dim=1, keepdims=True) + epsilon
    scl_neg = -torch.sum(Wt_neg, dim=1, keepdims=True) + epsilon
                                                                                                                                                                                                                                                                                                            
    # Leverage Constriant 
    ## If the long/short < 0.8, then set it to 0.9
    lev_constr = torch.maximum(scl_pos/scl_neg, thres[0]) # [T,N,1]
    ## If the long/short >1.2, then set it to 1.1
    lev_constr = torch.minimum(lev_constr, thres[1]) # [T,N,1]
                                                                                                                                                                                                                                                                                                            
    # Transform 
    ## First we set the negative leverage to 1 every day
    Wt_neg = Wt_neg / scl_neg
    ## Then we set the positive leverage to 1 and then multiply it by the lev_contr,
    ## which is in [0.8, 1.2]
    Wt_pos = lev_constr * Wt_pos / scl_pos
    Wt = Wt_neg + Wt_pos                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                        
    return Wt 

def Sharpe_loss(preds,ret,batch_size):
    weight_mat=preds.reshape(batch_size, -1)
    return_mat=ret.reshape(batch_size, -1)
    weight_mat=torch_get_adjusted_weight(weight_mat)
    rp = torch.sum(weight_mat * return_mat, dim=1)  
    rp_mean = torch.mean(rp)
    rp_std = torch.std(rp)
    annual_sr = np.sqrt(250) * rp_mean / rp_std
    loss = - rp_mean / rp_std 
    return loss

def MSEIC_loss(preds,ret,mask,batch_size):
    weight_mat=preds.reshape(batch_size, -1)
    return_mat=ret.reshape(batch_size, -1)
    mask_mat=mask.reshape(batch_size, -1)
    denominator = torch.sum(mask_mat)

    # rp = torch.sum(weight_mat * return_mat, dim=1)  
    # rp_mean = torch.mean(rp)
    # rp_std = torch.std(rp)
    # annual_sr = np.sqrt(250) * rp_mean / rp_std

    # MSE Loss
    mse_loss = torch.sum((weight_mat - return_mat)**2) / denominator
    # Correlation Loss
    y_true = return_mat - torch.sum(return_mat, dim=(0, 1), keepdim=True) / denominator
    y_pred = weight_mat - torch.sum(weight_mat, dim=(0, 1), keepdim=True) / denominator
    cov = torch.sum(y_true * y_pred)
    std_prod = torch.sqrt(torch.sum(y_true**2) * torch.sum(y_pred**2))
    corr_loss = - cov / (std_prod + epsilon)
    
    # Loss
    loss = ANN_CORR_WEIGHT * corr_loss + (1-ANN_CORR_WEIGHT) * mse_loss

    return loss

