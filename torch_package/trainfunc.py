'''
Description: This module is to define those functions we would like to use when preprocessing or evaluating the data.
             Most of the functions are under the numpy and pandas context.

Dependency:  numpy, tensorflow

Version:  2.0 or newer

Date: 2022.03.05

Contributor: Jialiang Lou, Keyu Zhou
'''
import numpy as np
import torch

from torch_package import functions, lossfunc, models

#=================================================================================#
#================================ Import Data ==================================#
#=================================================================================#
def train(model, model_class, st_ret_index, ndate, look_back, regu_how, regu_rate, CMat, RMat, IMat, criterion, optimizer, scheduler, device, \
            if_qunt, if_demean, if_scale, if_softmax, if_adv_thres, qunt_list, adv_mat, adv_thres):
    model.train() # Turn on training mode
    loss_list = []
    sr_list = []
    ret_list = []
    std_list = []
    batch_size_list = []

    rand_index = torch.randperm(len(st_ret_index))

    for i in rand_index:

        # Get start and end index
        st_ret_index_i = st_ret_index[i]
        if i == len(st_ret_index) - 1:
            ed_ret_index_i = ndate - 1
        else:
            ed_ret_index_i = st_ret_index[i+1] - 1
        batch_size = ed_ret_index_i - st_ret_index_i + 1
    
        # Input
        CMat_i = CMat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:,:].to(device)
        RMat_i = RMat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:].to(device)
        IMat_i = IMat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:].to(device)
        adv_i = adv_mat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:].to(device)
        # [t, N, K]
        # [
        # Get Mask 
        if model_class == models.TransformerModel:
            Mask = models.generate_lookback_mask(batch_size + (look_back - 1), look_back).to(device)
        
        # Get Loss
        if model_class == models.TransformerModel:
            weights = torch.squeeze(model(CMat_i, Mask), dim=-1)[look_back-1:,:]
        else:
            weights = torch.squeeze(model(CMat_i), dim=-1)[look_back-1:,:]
        
        # Further Transformation
        if if_qunt:
            weights = functions.torch_get_qunt_weight(weights, q=qunt_list.to(device))        
        if if_demean:
            weights = functions.torch_get_demean_weight(weights)
        if if_scale:
            weights = functions.torch_get_adjusted_weight(weights, thres=[1, 1])
        if if_softmax:
            weights = functions.torch_get_ls_softmax_weight(input_w=weights)
        if if_adv_thres: 
            weights = functions.torch_get_advthres_weight(input_w=weights, adv=adv_i[look_back - 1:, :], threshold=adv_thres)

        loss, ret, stdev, sr = criterion(weights, RMat_i[look_back-1:,:], IMat_i[look_back-1:,:])
        loss += lossfunc.get_regular_loss(model=model, how=regu_how, rate=regu_rate)

        # Gradient Derivation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Save
        loss_list.append(loss.item())
        ret_list.append(ret.item())
        std_list.append(stdev.item())
        sr_list.append(sr.item())
        batch_size_list.append(batch_size)

    avg_loss = np.average(loss_list, weights=batch_size_list)
    avg_ret = np.average(ret_list, weights=batch_size_list)
    avg_std = np.average(std_list, weights=batch_size_list)
    avg_sr = np.average(sr_list, weights=batch_size_list)
    
    return {'loss':avg_loss, 'ret':avg_ret, 'std':avg_std, 'sr':avg_sr, 'lr': scheduler.get_last_lr()[0]}

def evaluate(model, model_class, st_ret_index, ndate, look_back, CMat, RMat, IMat, criterion, device, \
                if_qunt, if_demean, if_scale, if_softmax, if_adv_thres, qunt_list, adv_mat, adv_thres):
    model.eval()  # turn on evaluation mode
    rand_index = [i for i in range(len(st_ret_index))]
    
    weights_v = []
    with torch.no_grad():
        for i in rand_index:

            # Get start and end index
            st_ret_index_i = st_ret_index[i]
            if i == len(st_ret_index) - 1:
                ed_ret_index_i = CMat.shape[0] - 1
            else:
                ed_ret_index_i = st_ret_index[i+1] - 1
            
            # Input
            CMat_i = CMat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:,:].to(device)
            IMat_i = IMat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:].to(device)
            
            # Get Mask
            if model_class == models.TransformerModel:
                batch_size = IMat_i.shape[0]
                Mask = models.generate_lookback_mask(batch_size, look_back).to(device)
            
            # Get Loss
            if model_class == models.TransformerModel:
                weights = torch.squeeze(model(CMat_i, Mask),dim=-1)[look_back-1:,:]
            else:
                weights = torch.squeeze(model(CMat_i),dim=-1)[look_back-1:,:]

            weights_v.append(weights)
        weights_v = torch.concat(weights_v, dim=0)[-ndate:,:]

        # Further Transformation
        if if_qunt:
            weights_v = functions.torch_get_qunt_weight(weights_v, q=qunt_list.to(device))        
        if if_demean:
            weights_v = functions.torch_get_demean_weight(weights_v)
        if if_scale:
            weights_v = functions.torch_get_adjusted_weight(weights_v, thres=[1, 1])
        if if_softmax:
            weights_v = functions.torch_get_ls_softmax_weight(input_w=weights_v)
        if if_adv_thres: 
            weights_v = functions.torch_get_advthres_weight(input_w=weights_v, adv=adv_mat.to(device), threshold=adv_thres)

        loss, *_ = criterion(weights_v, RMat[-ndate:,:].to(device), IMat[-ndate:,:].to(device))                

    return {'loss':loss.cpu().numpy()}

def predict(model, model_class, st_ret_index, ndate, look_back, CMat, device, \
            if_qunt, if_demean, if_scale, if_softmax, if_adv_thres, qunt_list, adv_mat, adv_thres):
    model.eval()  # turn on evaluation mode
    rand_index = [i for i in range(len(st_ret_index))]

    weights_p = []
    with torch.no_grad():
        for i in rand_index:

            # Get start and end index
            st_ret_index_i = st_ret_index[i]
            if i == len(st_ret_index) - 1:
                ed_ret_index_i = CMat.shape[0] - 1
            else:
                ed_ret_index_i = st_ret_index[i+1] - 1
            
            # Input
            CMat_i = CMat[st_ret_index_i-(look_back-1):ed_ret_index_i+1,:,:].to(device)

            # Get Mask
            batch_size = CMat_i.shape[0]
            if model_class == models.TransformerModel:
                Mask = models.generate_lookback_mask(batch_size, look_back).to(device)
            
            # Get Loss
            if model_class == models.TransformerModel:
                weights = torch.squeeze(model(CMat_i, Mask), dim=-1)[look_back-1:,:]
            else:
                weights = torch.squeeze(model(CMat_i), dim=-1)[look_back-1:,:]
            weights_p.append(weights)
            
        weights_p = torch.concat(weights_p, dim=0)[-ndate:,:]
        
        # Further Transformation
        if if_qunt:
            weights_p = functions.torch_get_qunt_weight(weights_p, q=qunt_list.to(device))        
        if if_demean:
            weights_p = functions.torch_get_demean_weight(weights_p)
        if if_scale:
            weights_p = functions.torch_get_adjusted_weight(weights_p, thres=[1, 1])
        if if_softmax:
            weights_p = functions.torch_get_ls_softmax_weight(input_w=weights_p)
        if if_adv_thres: 
            weights_p = functions.torch_get_advthres_weight(input_w=weights_p, adv=adv_mat.to(device), threshold=adv_thres)

    return {'weights':weights_p}
            