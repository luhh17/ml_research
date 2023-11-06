# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import data_organization
import torch
import pickle as pkl
import sys

def split_signal_return_mask(data_array, columns, data_mask,ret_var,target_window):
    """
    This function splits the data array into signal matrix, return matrix, multi_return_matrix and mask matrix.
    """
    # the first three columns are related to returns
    input_CMat_t = np.nan_to_num(data_array[..., 3:], copy=False)
    input_RMat_t = data_array[..., columns.get_loc(ret_var)]
    input_IMat_t = data_mask.astype('bool')
    # calculate multi-day return as target according to target_window
    input_RMat_t_multid=(pd.DataFrame(input_RMat_t).T.rolling(target_window).sum().shift(-(target_window-1))).fillna(0).T.to_numpy().astype('float32')
    input_CMat_t, input_RMat_t,input_multiRMat_t,input_IMat_t = torch.from_numpy(input_CMat_t), torch.from_numpy(
    input_RMat_t),torch.from_numpy(
    input_RMat_t_multid), torch.from_numpy(input_IMat_t)

    return input_CMat_t, input_RMat_t, input_multiRMat_t,input_IMat_t

def top2000_contruct_train_valid(data_path,file_name,year,ret_var,target_window):
    """
    This function constructs the training and validation sets for the top 2000 companies based on market capitalization.
    
    Returns:
    - Tuple: A tuple containing the training and validation sets.
    """
    train_start_date=f'{year - 3}-01-01'
    train_end_date = f'{year - 1}-08-31'
    valid_start_date = f'{year - 1}-09-01'
    valid_end_date = f'{year - 1}-12-31'

    fea_index=pd.read_pickle(data_path+'/'+f'index.pkl')
    sel_index=fea_index[(fea_index['date']>=train_start_date) & (fea_index['date']<=valid_end_date)].index
    data=pd.read_hdf(data_path + '/'+file_name, start=sel_index[0], stop=sel_index[-1]+1)
    

    # No extreme return
    data_filter = data[np.abs(data['ret_open'])<0.25].reset_index(drop=True)
    print(data_filter.shape)
    del data

    # top 2000 universe
    data_filter['year'] = data_filter['date'].dt.year
    data_filter_top2000 = pd.DataFrame()
    for year in data_filter['year'].unique():
        year_data = data_filter[data_filter['year'] == year]
        first_date_of_year = year_data['date'].min()
        year_data_first_date = year_data[year_data['date'] == first_date_of_year].copy()
        year_data_first_date.loc[:,'mktcap_r'] = year_data_first_date['C.1.1'].rank(ascending=False, method='min')
        top2000_universe=year_data_first_date[year_data_first_date['mktcap_r'] <= 2000]['stkcd']
        data_filter_top2000 = pd.concat([data_filter_top2000, year_data[year_data['stkcd'].isin(top2000_universe)]])
        del year_data,year_data_first_date

    del data_filter_top2000['year']
    del data_filter
    print(data_filter_top2000.shape)

    data_filter_top2000.set_index(['stkcd', 'date'], inplace=True)
    train_array, train_mask, _, train_columns, date_matrix, stkcd_matrix = data_organization.convert_array(data_filter_top2000,ret_var)
    signal_matrix, return_matrix, multireturn_matrix,mask_matrix = split_signal_return_mask(train_array, train_columns, train_mask,ret_var,target_window)

    date_matrix = date_matrix.squeeze()
    stkcd_matrix = np.expand_dims(stkcd_matrix, axis=1)
    stkcd_matrix = np.tile(stkcd_matrix, (1, signal_matrix.shape[1]))
    print('signal_matrix.shape', signal_matrix.shape)
    print('return_matrix.shape', return_matrix.shape)
    print('mask_matrix.shape', mask_matrix.shape)
    print('date_matrix.shape', date_matrix.shape)
    print('stkcd_matrix.shape', stkcd_matrix.shape)

    date_list = date_matrix[0]
    # train set
    train_idx = (date_list >= int(train_start_date.replace('-', ''))) & (date_list <= int(train_end_date.replace('-', '')))
    train_signal = signal_matrix[:, train_idx, :]
    train_ret = return_matrix[:, train_idx]
    train_multiret=multireturn_matrix[:, train_idx]
    train_mask = mask_matrix[:, train_idx]
    train_date = date_matrix[:, train_idx]
    train_stkcd = stkcd_matrix[:, train_idx]
    print(train_signal.shape)
    print(train_ret.shape)
    print(train_mask.shape)
    print(train_date.shape)
    print(train_stkcd.shape)
    # validation set
    valid_idx = (date_list >= int(valid_start_date.replace('-', ''))) & (date_list <= int(valid_end_date.replace('-', '')))
    valid_signal = signal_matrix[:, valid_idx, :]
    valid_ret = return_matrix[:, valid_idx]
    valid_multiret=multireturn_matrix[:, valid_idx]
    valid_mask = mask_matrix[:, valid_idx]
    valid_date = date_matrix[:, valid_idx]
    valid_stkcd = stkcd_matrix[:, valid_idx]
    print(valid_signal.shape)
    print(valid_ret.shape)
    print(valid_mask.shape)
    print(valid_date.shape)
    print(valid_stkcd.shape)

    del signal_matrix,return_matrix,multireturn_matrix,mask_matrix,date_matrix,stkcd_matrix

    return (train_signal, train_ret, train_multiret,train_mask, train_date, train_stkcd),(valid_signal, valid_ret, valid_multiret,valid_mask, valid_date, valid_stkcd)

def total_universe_contruct_valid_test(data_path,file_name,year,ret_var,target_window):
    """
    This function constructs the validation and test sets for the top universe.
    
    Returns:
    - Tuple: A tuple containing the validation and test sets.
    """
    valid_start_date = f'{year - 1}-09-01'
    valid_end_date = f'{year - 1}-12-31'
    test_start_date = f'{year}-01-01'
    test_end_date = f'{year}-12-31'

    fea_index=pd.read_pickle(data_path+'/'+f'index.pkl')
    sel_index=fea_index[(fea_index['date']>=valid_start_date) & (fea_index['date']<=test_end_date)].index
    data=pd.read_hdf(data_path + '/'+file_name, start=sel_index[0], stop=sel_index[-1]+1)
    
    # No extreme return
    data_filter = data[np.abs(data['ret_open'])<0.25].reset_index(drop=True)
    print(data_filter.shape)
    del data
    
    data_filter.set_index(['stkcd', 'date'], inplace=True)
    train_array, train_mask, _, train_columns, date_matrix, stkcd_matrix = data_organization.convert_array(data_filter,ret_var)
    signal_matrix, return_matrix,multireturn_matrix, mask_matrix = split_signal_return_mask(train_array, train_columns, train_mask,ret_var,target_window)

    date_matrix = date_matrix.squeeze()
    stkcd_matrix = np.expand_dims(stkcd_matrix, axis=1)
    stkcd_matrix = np.tile(stkcd_matrix, (1, signal_matrix.shape[1]))
    print('signal_matrix.shape', signal_matrix.shape)
    print('return_matrix.shape', return_matrix.shape)
    print('mask_matrix.shape', mask_matrix.shape)
    print('date_matrix.shape', date_matrix.shape)
    print('stkcd_matrix.shape', stkcd_matrix.shape)

    date_list = date_matrix[0]
    # validation set
    valid_idx = (date_list >= int(valid_start_date.replace('-', ''))) & (date_list <= int(valid_end_date.replace('-', '')))
    valid_signal = signal_matrix[:, valid_idx, :]
    valid_ret = return_matrix[:, valid_idx]
    valid_multiret=multireturn_matrix[:, valid_idx]
    valid_mask = mask_matrix[:, valid_idx]
    valid_date = date_matrix[:, valid_idx]
    valid_stkcd = stkcd_matrix[:, valid_idx]
    print(valid_signal.shape)
    print(valid_ret.shape)
    print(valid_mask.shape)
    print(valid_date.shape)
    print(valid_stkcd.shape)
    # test set
    test_idx = (date_list >= int(test_start_date.replace('-', ''))) & (date_list <= int(test_end_date.replace('-', '')))
    test_signal = signal_matrix[:, test_idx, :]
    test_ret = return_matrix[:, test_idx]
    test_multiret=multireturn_matrix[:, test_idx]
    test_mask = mask_matrix[:, test_idx]
    test_date = date_matrix[:, test_idx]
    test_stkcd = stkcd_matrix[:, test_idx]
    print(test_signal.shape)
    print(test_ret.shape)
    print(test_mask.shape)
    print(test_date.shape)
    print(test_stkcd.shape)

    del signal_matrix,return_matrix,multireturn_matrix,mask_matrix,date_matrix,stkcd_matrix

    return (valid_signal, valid_ret,valid_multiret, valid_mask, valid_date, valid_stkcd),(test_signal, test_ret,test_multiret, test_mask, test_date, test_stkcd)


def return_divide_train_valid(train_data, valid_data, window_length):
    """
    Calculate the lag 1 return for train and valid sets.
    First consider the window length issue, then shift the return to get the lag 1 return.
    """

    test_start_idx = train_data.shape[1] + valid_data.shape[1]
    valid_start_idx = train_data.shape[1]

    new_data = np.concatenate([train_data, valid_data], axis=1)
    new_train = new_data[:, :valid_start_idx]
    new_valid = new_data[:, valid_start_idx - window_length + 1:test_start_idx]

    test_start_idx = new_train.shape[1] + new_valid.shape[1]
    valid_start_idx = new_train.shape[1]

    new_data = np.concatenate([new_train, new_valid], axis=1)
    new_train = new_data[:, :valid_start_idx-1]
    new_train = (np.concatenate([np.zeros((new_train.shape[0],1)), new_train], axis=1)).astype('float32')
    new_valid = new_data[:, valid_start_idx-1 : test_start_idx-1]

    return new_train, new_valid

def return_divide_test(valid_data, test_data,window_length):
    """
    Calculate the lag 1 return for test sets.
    Simultaneously consider the window length issue and the shift issue.
    """
    test_start_idx =valid_data.shape[1]
    new_data = np.concatenate([valid_data, test_data], axis=1)
    new_test = new_data[:, test_start_idx - window_length:-1]
    return new_test

def divide_train_valid(train_data, valid_data,window_length):
    """
    Revision on the train and valid sets with respect to window_length.
    """
    test_start_idx = train_data.shape[1] + valid_data.shape[1]
    valid_start_idx = train_data.shape[1]
    new_data = np.concatenate([train_data, valid_data], axis=1)
    new_train = new_data[:, :valid_start_idx]
    new_valid = new_data[:, valid_start_idx - window_length + 1:test_start_idx]
    return new_train, new_valid

def divide_test(valid_data, test_data,window_length):
    """
    Revision on the test sets with respect to window_length.
    Write two functions for the sake of different universe, the train and valid contains only top2000 but the test set contains all stocks.
    """
    test_start_idx =valid_data.shape[1]
    new_data = np.concatenate([valid_data, test_data], axis=1)
    new_test = new_data[:, test_start_idx - window_length + 1:]
    return new_test

if __name__ == '__main__':
    data_path = '/mnt/HDD16TB/output_20231015'
    output_data_path='/mnt/HDD16TB/data_output_20231015'
    file_name='signal_rank.h5'
    ret_var='exret_open'
    year=int(sys.argv[1])
    target_window=int(sys.argv[2])
    window_length=int(sys.argv[3])
    print(year)

    train_data,valid_data=top2000_contruct_train_valid(data_path,file_name,year,ret_var,target_window) #top2000 universe
    full_valid_data,full_test_data=total_universe_contruct_valid_test(data_path,file_name,year,ret_var,target_window) #total universe

    train_signal, train_1dret, train_multiret,train_mask, train_time, train_stkcd = train_data
    valid_signal, valid_1dret, valid_multiret,valid_mask, valid_time, valid_stkcd = valid_data
    full_valid_signal, full_valid_1dret, full_valid_multiret,full_valid_mask, full_valid_time, full_valid_stkcd = full_valid_data
    full_test_signal, full_test_1dret, full_test_multiret,full_test_mask, full_test_time, full_test_stkcd = full_test_data

    fea_num=train_signal.shape[2]
    del train_data,valid_data,full_valid_data,full_test_data

    if window_length>1:
        train_signal, valid_signal = divide_train_valid(train_signal, valid_signal,window_length)
        full_test_signal = divide_test(full_valid_signal, full_test_signal,window_length)

        train_multiret, valid_multiret = divide_train_valid(train_multiret, valid_multiret,window_length)
        full_test_multiret = divide_test(full_valid_multiret, full_test_multiret,window_length)

        train_mask, valid_mask = divide_train_valid(train_mask, valid_mask,window_length)
        full_test_mask = divide_test(full_valid_mask, full_test_mask,window_length)  

        train_time, valid_time = divide_train_valid(train_time, valid_time,window_length)
        full_test_time = divide_test(full_valid_time, full_test_time,window_length)

        train_stkcd, valid_stkcd = divide_train_valid(train_stkcd, valid_stkcd,window_length)
        full_test_stkcd = divide_test(full_valid_stkcd, full_test_stkcd,window_length)    

    # construct the lag 1 return, note that, we use 1dret to construct the lag 1 return,because if we use the multi-day return, this will lead to future info leakage problem.
    train_lret, valid_lret=return_divide_train_valid(train_1dret, valid_1dret,window_length)
    full_test_lret = return_divide_test(full_valid_1dret, full_test_1dret,window_length)

    train = (train_signal, train_multiret, train_lret, train_mask, train_time, train_stkcd)
    valid = (valid_signal, valid_multiret, valid_lret, valid_mask, valid_time, valid_stkcd)
    test = (full_test_signal, full_test_multiret, full_test_lret, full_test_mask, full_test_time, full_test_stkcd)

    if target_window==1:
        with open(f'{output_data_path}/roll_{year}_{ret_var}_{window_length}_balanced.pkl', 'wb') as file1:
            pkl.dump((train, valid, test,fea_num), file1)
    else:
        with open(f'{output_data_path}/roll_{year}_{ret_var}_{target_window}d_{window_length}_balanced.pkl', 'wb') as file1:
            pkl.dump((train, valid, test,fea_num), file1)

