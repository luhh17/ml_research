'''
Description: This module is to define those functions we would like to use when preprocessing or evaluating the data.
             Most of the functions are under the numpy and pandas context.

Dependency:  numpy, tensorflow

Version:  2.0 or newer

Date: 2022.03.05

Contributor: Jialiang Lou, Keyu Zhou
'''
import numpy as np
import pandas as pd
import torch
import torch_package.functions as functions
import torch_package.models as models

#=================================================================================#
#================================ Import Data ==================================#
#=================================================================================#
# Import Data Full Function
def get_tvp_datasets(model_class, oos_date, test_len, gap_len, valid_len, train_len, \
                    train_index_df, valid_index_df, test_index_df, train_data_file, valid_data_file, test_data_file, if_dummy, dummy_list):

    test_start = oos_date
    test_end = functions.calculate_year_month(oos_date, test_len - 1)
    val_end = functions.calculate_year_month(test_start, -1 - gap_len)
    val_start = functions.calculate_year_month(val_end, -valid_len + 1)
    train_end = functions.calculate_year_month(val_start, -1)
    train_start = functions.calculate_year_month(train_end, -train_len + 1)

    if model_class == models.TransformerModel:
        test_start_tfm = functions.calculate_year_month(oos_date, -1)
        val_start_tfm = functions.calculate_year_month(val_end, -valid_len + 1 - 1)
    

    train_idx_start, train_idx_end = functions.get_start_end_index(train_index_df['date'], train_start, train_end)
    if model_class == models.TransformerModel:
        val_idx_start_tfm, val_idx_end_tfm = functions.get_start_end_index(valid_index_df['date'], val_start_tfm, val_end)
        test_idx_start_tfm, test_idx_end_tfm = functions.get_start_end_index(test_index_df['date'], test_start_tfm, test_end)
    val_idx_start, val_idx_end = functions.get_start_end_index(valid_index_df['date'], val_start, val_end)
    test_idx_start, test_idx_end = functions.get_start_end_index(test_index_df['date'], test_start, test_end)

    # Read data
    train_data = pd.read_hdf(train_data_file, start=train_idx_start, stop=train_idx_end)
    d_input = train_data.shape[1] - 4
    if model_class == models.TransformerModel:
        val_data = pd.read_hdf(valid_data_file, start=val_idx_start_tfm, stop=val_idx_end_tfm)
        test_data = pd.read_hdf(test_data_file, start=test_idx_start_tfm, stop=test_idx_end_tfm)     
    else:
        val_data = pd.read_hdf(valid_data_file, start=val_idx_start, stop=val_idx_end)
        test_data = pd.read_hdf(test_data_file, start=test_idx_start, stop=test_idx_end)
    ## Drop Dummmy or not
    if if_dummy == False:
        train_data.drop(columns=dummy_list, inplace=True)
        val_data.drop(columns=dummy_list, inplace=True)
        test_data.drop(columns=dummy_list, inplace=True)

    # Number of Date
    ndate_train = len(train_data['date'].unique())
    if model_class == models.TransformerModel:
        ndate_valid = len(pd.read_hdf(valid_data_file, start=val_idx_start, stop=val_idx_end)['date'].unique())
        ndate_test = len(pd.read_hdf(test_data_file, start=test_idx_start, stop=test_idx_end)['date'].unique())
    else:    
        ndate_valid = len(val_data['date'].unique())
        ndate_test = len(test_data['date'].unique())

    return train_data, val_data, test_data, d_input, [ndate_train, ndate_valid, ndate_test]

def get_datasets(date_start, date_len, index_df, data_file, if_dummy, dummy_list):
    
    date_start = date_start
    date_end = functions.calculate_year_month(date_start, date_len - 1)
    idx_start, idx_end = functions.get_start_end_index(index_df['date'], date_start, date_end)

    # Read data
    data = pd.read_hdf(data_file, start=idx_start, stop=idx_end)
    d_input = data.shape[1] - 4

    ## Drop Dummmy or not
    if if_dummy == False:
        data.drop(columns=dummy_list, inplace=True)
        
    # Number of Date
    ndate = len(data['date'].unique())
    
    return data, d_input, ndate
    
# Matrices Transformation Full Function
def get_CIR_Mats(train_data, val_data, test_data, date_var, stock_var, ret_type, d_input):
    
    # Date and stock Dictionary
    TDict_t, NDict_t = functions.get_tn_dicts(train_data, date_var, stock_var)
    TDict_v, NDict_v = functions.get_tn_dicts(val_data, date_var, stock_var)
    TDict_p, NDict_p = functions.get_tn_dicts(test_data, date_var, stock_var)

    # Convert to arrays
    ## Set index
    train_data.set_index([date_var, stock_var], inplace=True)
    val_data.set_index([date_var, stock_var], inplace=True)
    test_data.set_index([date_var, stock_var], inplace=True)
    
    ## Get matrices
    train_array, train_mask, _, train_columns = functions.convert_array(train_data)
    val_array, val_mask, _, val_columns = functions.convert_array(val_data)
    test_array, test_mask, _, test_columns = functions.convert_array(test_data)
    
    input_CMat_t = np.nan_to_num(train_array[..., -d_input:-2], copy=False)
    input_RMat_t = train_array[..., train_columns.get_loc(ret_type)]
    input_IMat_t = train_mask.astype(input_CMat_t.dtype)
    input_ADVMat_t = train_array[..., train_columns.get_loc('adv_20')]
    input_CMat_t, input_RMat_t, input_IMat_t, input_ADVMat_t = torch.from_numpy(input_CMat_t), torch.from_numpy(input_RMat_t), torch.from_numpy(input_IMat_t), torch.from_numpy(input_ADVMat_t)
    
    input_CMat_v = np.nan_to_num(val_array[..., -d_input:-2], copy=False)
    input_RMat_v = val_array[..., val_columns.get_loc(ret_type)]
    input_IMat_v = val_mask.astype(input_CMat_v.dtype)
    input_ADVMat_v = val_array[..., val_columns.get_loc('adv_20')]
    input_CMat_v, input_RMat_v, input_IMat_v, input_ADVMat_v = torch.from_numpy(input_CMat_v), torch.from_numpy(input_RMat_v), torch.from_numpy(input_IMat_v), torch.from_numpy(input_ADVMat_v)
    
    input_CMat_p = np.nan_to_num(test_array[..., -d_input:-2], copy=False)
    input_RMat_p = test_array[..., test_columns.get_loc(ret_type)]
    input_IMat_p = test_mask.astype(input_CMat_p.dtype)
    input_ADVMat_p = test_array[..., test_columns.get_loc('adv_20')]
    input_CMat_p, input_RMat_p, input_IMat_p, input_ADVMat_p = torch.from_numpy(input_CMat_p), torch.from_numpy(input_RMat_p), torch.from_numpy(input_IMat_p), torch.from_numpy(input_ADVMat_p)

    return  TDict_t, NDict_t, TDict_v, NDict_v, TDict_p, NDict_p, \
            input_CMat_t, input_RMat_t, input_IMat_t, input_ADVMat_t, \
            input_CMat_v, input_RMat_v, input_IMat_v, input_ADVMat_v, \
            input_CMat_p, input_RMat_p, input_IMat_p, input_ADVMat_p


def get_Mats(data, date_var, stock_var, ret_type, d_input):
    
    # Date and stock Dictionary
    TDict, NDict = functions.get_tn_dicts(data, date_var, stock_var)

    # Convert to arrays
    ## Set index
    data.set_index([date_var, stock_var], inplace=True)
    
    ## Get matrices
    array, mask, _, columns = functions.convert_array(data)
    
    CMat = np.nan_to_num(array[..., -d_input:-2], copy=False)
    RMat = array[..., columns.get_loc(ret_type)]
    IMat = mask.astype(CMat.dtype)
    ADVMat = array[..., columns.get_loc('adv_20')]
    CMat, RMat, IMat, ADVMat = torch.from_numpy(CMat), torch.from_numpy(RMat), torch.from_numpy(IMat), torch.from_numpy(ADVMat)

    return TDict, NDict, CMat, RMat, IMat, ADVMat
            

    
    