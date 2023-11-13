# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import prepare_dataset.data_organization
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

def contruct_train_valid_test(data_path, file_name, sample_start_date: str, sample_end_date: str, 
    ret_var: str, target_window, top_n: int=2000):
    """
    This function constructs the training and validation sets for the top 2000 companies based on market capitalization.
    
    Returns:
    - Tuple: A tuple containing the training and validation sets.
    """

    fea_index = pd.read_pickle(data_path+'/'+f'index.pkl')
    sel_index = fea_index[(fea_index['date']>=train_start_date) & (fea_index['date']<=valid_end_date)].index
    data = pd.read_hdf(data_path + '/'+file_name, start=sel_index[0], stop=sel_index[-1]+1)
    

    # No extreme return
    data_filter = data[np.abs(data['ret_open'])<0.25].reset_index(drop=True)
    del data

    # top 2000 universe
    data_filter['year'] = data_filter['date'].dt.year
    data_filter_top2000 = pd.DataFrame()
    for year in data_filter['year'].unique():
        year_data = data_filter[data_filter['year'] == year]
        first_date_of_year = year_data['date'].min()
        year_data_first_date = year_data[year_data['date'] == first_date_of_year].copy()
        year_data_first_date.loc[:,'mktcap_r'] = year_data_first_date['C.1.1'].rank(ascending=False, method='min')
        topn_universe=year_data_first_date[year_data_first_date['mktcap_r'] <= top_n]['stkcd']
        data_filter_top2000 = pd.concat([data_filter_top2000, year_data[year_data['stkcd'].isin(topn_universe)]])
        del year_data, year_data_first_date

    del data_filter_top2000['year']
    del data_filter
    # print(data_filter_top2000.shape)

    data_filter_top2000.set_index(['stkcd', 'date'], inplace=True)
    train_array, train_mask, _, train_columns, date_matrix, stkcd_matrix = data_organization.convert_array(data_filter_top2000, ret_var)
    signal_matrix, return_matrix, multireturn_matrix,mask_matrix = split_signal_return_mask(train_array, train_columns, train_mask,ret_var, target_window)

    date_matrix = date_matrix.squeeze()
    stkcd_matrix = np.expand_dims(stkcd_matrix, axis=1)
    stkcd_matrix = np.tile(stkcd_matrix, (1, signal_matrix.shape[1]))

    date_list = date_matrix[0]
    # train set
    train_idx = (date_list >= int(sample_start_date.replace('-', ''))) & (date_list <= int(sample_end_date.replace('-', '')))
    train_signal = signal_matrix[:, train_idx, :]
    train_ret = return_matrix[:, train_idx]
    train_multiret=multireturn_matrix[:, train_idx]
    train_mask = mask_matrix[:, train_idx]
    train_date = date_matrix[:, train_idx]
    train_stkcd = stkcd_matrix[:, train_idx]

    # # validation set
    # valid_idx = (date_list >= int(valid_start_date.replace('-', ''))) & (date_list <= int(valid_end_date.replace('-', '')))
    # valid_signal = signal_matrix[:, valid_idx, :]
    # valid_ret = return_matrix[:, valid_idx]
    # valid_multiret=multireturn_matrix[:, valid_idx]
    # valid_mask = mask_matrix[:, valid_idx]
    # valid_date = date_matrix[:, valid_idx]
    # valid_stkcd = stkcd_matrix[:, valid_idx]

    # test_idx = (date_list >= int(test_start_date.replace('-', ''))) & (date_list <= int(test_end_date.replace('-', '')))
    # test_signal = signal_matrix[:, test_idx, :]
    # test_ret = return_matrix[:, test_idx]
    # test_multiret = multireturn_matrix[:, test_idx]
    # test_mask = mask_matrix[:, test_idx]
    # test_date = date_matrix[:, test_idx]
    # test_stkcd = stkcd_matrix[:, test_idx]


    del signal_matrix, return_matrix, multireturn_matrix, mask_matrix, date_matrix, stkcd_matrix

    return train_signal, train_ret, train_multiret, train_mask, train_date, train_stkcd
     



def return_divide_train_valid_test(train_data, valid_data, test_data, window_length):
    """
    Calculate the lag 1 return for train and valid sets.
    First consider the window length issue, then shift the return to get the lag 1 return.
    """

    test_start_idx = train_data.shape[1] + valid_data.shape[1]
    valid_start_idx = train_data.shape[1]

    new_data = np.concatenate([train_data, valid_data, test_data], axis=1)
    new_train = new_data[:, :valid_start_idx]
    new_valid = new_data[:, valid_start_idx - window_length + 1: test_start_idx]
    new_test = new_data[:, test_start_idx - window_length + 1:]

    test_start_idx = new_train.shape[1] + new_valid.shape[1]
    valid_start_idx = new_train.shape[1]

    new_data = np.concatenate([new_train, new_valid, new_test], axis=1)
    new_train = new_data[:, :valid_start_idx-1]
    new_train = (np.concatenate([np.zeros((new_train.shape[0],1)), new_train], axis=1)).astype('float32')
    new_valid = new_data[:, valid_start_idx-1 : test_start_idx-1]
    new_test = new_data[:, test_start_idx-1: -1]

    return new_train, new_valid, new_test

# def return_divide_test(valid_data, test_data,window_length):
#     """
#     Calculate the lag 1 return for test sets.
#     Simultaneously consider the window length issue and the shift issue.
#     """
#     test_start_idx =valid_data.shape[1]
#     new_data = np.concatenate([valid_data, test_data], axis=1)
#     new_test = new_data[:, test_start_idx - window_length:-1]
#     return new_test


def reappend_train_valid_test(train_data: torch.Tensor, 
valid_data: torch.Tensor, 
test_data: torch.Tensor, 
window_length: int) -> tuple:
    """
    原始的train_data, valid_data, test_data是按照window_length = 1来划分的
    如果window_length > 1，那么为了确保valid和test的样本量不变，需要向前重新划分
    """
    test_start_idx = train_data.shape[1] + valid_data.shape[1]
    valid_start_idx = train_data.shape[1]
    new_data = np.concatenate([train_data, valid_data, test_data], axis=1)
    new_train = new_data[:, :valid_start_idx]
    new_valid = new_data[:, valid_start_idx - window_length + 1: test_start_idx]
    new_test = new_data[:, test_start_idx - window_length + 1:]
    return new_train, new_valid, new_test


def check_data_exist(args: dict, test_year_list: list[int]) -> None:
    """
    Check if data file exists, if not, create them
    """
    for test_year in test_year_list:
        path = f"{args['output_dataset_path']}/roll_{test_year}_{args['ret_type']}_{args['target_window']}d_{args['window_length']}_balanced.pkl"
        if not os.path.exists(path):
            construct_dataset(args, test_year)
    return



def construct_dataset(args, test_year):
    data_path = args['raw_data_path']
    output_data_path = args['output_dataset_path']
    file_name = 'signal_rank.h5'
    ret_var = args['ret_type']
    year = test_year
    test_start_date = f"{year}-01-01"
    test_end_date = f"{year}-12-31"
    valid_start_date = f"{year-1}-0{12-args['valid_window']+1}-01"
    valid_end_date = f"{year-1}-12-31"
    train_start_date = f"{year-args['train_window']}-01-01"
    end_date = "31" if (12-args['valid_window']) % 2 == 1 else "30"
    end_date = "28" if (12-args['valid_window']) == 2 else end_date
    train_end_date = f"{year-1}-0{12-args['valid_window']}-{end_date}"

    target_window = args['target_window']
    window_length = args['window_length']
    
    train_signal, train_1dret, train_multiret, train_mask, train_time, train_stkcd = contruct_train_valid_test(data_path, file_name, 
    train_start_date, train_end_date, ret_var, target_window, top_n=args['train_top_n']) 

    valid_signal, valid_1dret, valid_multiret, valid_mask, valid_time, valid_stkcd = contruct_train_valid_test(data_path, file_name, 
    valid_start_date, valid_end_date, ret_var, target_window, top_n=args['valid_top_n']) 

    test_signal, test_1dret, test_multiret, test_mask, test_time, test_stkcd = contruct_train_valid_test(data_path, file_name, 
    test_start_date, test_end_date, ret_var, target_window, top_n=args['test_top_n'])

    fea_num = train_signal.shape[2]

    if window_length > 1:
        train_signal, valid_signal, test_signal = reappend_train_valid_test(train_signal, valid_signal, test_signal, window_length)
       
        train_multiret, valid_multiret, test_multiret = reappend_train_valid_test(train_multiret, valid_multiret, test_multiret, window_length)

        train_mask, valid_mask, test_mask = reappend_train_valid_test(train_mask, valid_mask, test_mask, window_length) 

        train_time, valid_time, test_time = reappend_train_valid_test(train_time, valid_time, test_time, window_length)

        train_stkcd, valid_stkcd, test_stkcd = reappend_train_valid_test(train_stkcd, valid_stkcd, test_stkcd, window_length)

    # construct the lag 1 return, note that, we use 1dret to construct the lag 1 return,because if we use the multi-day return, this will lead to future info leakage problem.
    train_lret, valid_lret, test_lret = return_divide_train_valid_test(train_1dret, valid_1dret, test_1dret, window_length)
    # full_test_lret = return_divide_test(full_valid_1dret, full_test_1dret, window_length)

    train = (train_signal, train_multiret, train_lret, train_mask, train_time, train_stkcd)
    valid = (valid_signal, valid_multiret, valid_lret, valid_mask, valid_time, valid_stkcd)
    test = (test_signal, test_multiret, test_lret, test_mask, test_time, test_stkcd)


    with open(f'{output_data_path}/roll_{year}_{ret_var}_{target_window}d_{window_length}_balanced.pkl', 'wb') as file1:
        pkl.dump((train, valid, test,fea_num), file1)




if __name__ == '__main__':
    pass