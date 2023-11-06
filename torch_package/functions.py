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
import statsmodels.api as sm
import torch
from numba import njit

torch.manual_seed(0)
epsilon = 1e-12
#=================================================================================#
#================================ Import Data ==================================#
#=================================================================================#

def calculate_year_month(start, num_months, gap=0):
    'Calculate the year-month tuple for a given start year-month tuple and number of months'
    year, month = start
    month += (num_months + gap)
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year, month

def get_start_end_index(date_series, start_year_month, end_year_month):
    '''
    Return the start and end index of the date range

    Parameters
    ----------
    date_series : pd.Series
        A date series. It should have an integer index starting from 0.
    start_year_month : tuple[int, int]
        The start year and month.
    end_year_month : tuple[int, int]
        The end year and month.
    
    Returns
    -------
    start_index : int
        The start index.
    end_index : int
        The end index. Note that the end index is not included.
    '''
    start_ym = start_year_month[0] * 12 + start_year_month[1]
    end_ym = end_year_month[0] * 12 + end_year_month[1]
    ym = date_series.dt.year * 12 + date_series.dt.month
    index = date_series.index[(ym >= start_ym) & (ym <= end_ym)]
    return index[0], index[-1] + 1

def get_time_ym(train_len, valid_len, gap_len, test_len, start_year=None, start_ym=None):
    '''
    Description
    -----------
        Get the ym index of the periods we want.
    
    Parameters
    ----------
        start_year: Out-of-Sample start year
        test_len: Length of the training period in month.
        valid_len: Length of the validation period in month.
        gatest_len: Length of the gap period.
        test_len: Length of the testing period in month.

    Output
    ------
        training period: [start_ym_t, end_ym_t]
        validation period: [start_ym_v, end_ym_v]
        validation period: [start_ym_v, end_ym_v]
        
    '''
    # Start and End YM
    ## Testing Sample
    if start_ym is None:
        # The year-month calculation rule is from 1960, which means Jan1960 = 0
        start_ym_p = (start_year - 1960) * 12
    else:
        start_ym_p = start_ym
        
    end_ym_p = start_ym_p + test_len - 1
    test_period = [start_ym_p, end_ym_p]
    
    ## Validation Sample
    start_ym_v = start_ym_p - gap_len - valid_len
    end_ym_v = start_ym_p - gap_len - 1
    valid_period = [start_ym_v, end_ym_v]
    
    ## Training Sample
    start_ym_t = start_ym_v - train_len
    end_ym_t = start_ym_v - 1
    train_period = [start_ym_t, end_ym_t]
    
    # Min Max Year
    min_yr = int(start_ym_t/12) + 1960
    max_yr = int(end_ym_p/12) + 1960
    minmax_yr = [min_yr, max_yr]

    return train_period, valid_period, test_period, minmax_yr

#=================================================================================#
#=============================== Data Preprocess =================================#
#=================================================================================#

# Month dictionary
def get_month_dict(retdata, varname):
    '''
    Description
    -----------
        Get the month index dictionary.
    
    Parameters
    ----------
        retdata: pd.DataFrame
        varname: string, the name of the month variable
    
    Output
    ------
        MonthDict: dict[month, index]
        MonthList: list, all the month names
        
    '''
    
    MonthDict = {}
    for irow, month_i in enumerate(retdata[varname].values):
        if MonthDict.get(month_i) is None:
            MonthDict[month_i] = [irow]
        else:
            MonthDict[month_i].append(irow)
    MonthList = sorted(MonthDict.keys())
    return MonthDict, MonthList

# Category dictionary
def get_cate_dict(data, idx, category):
    CateDict = {}
    for cate, i in zip(data[category].values, data[idx].values):
        if CateDict.get(cate) is None:
            CateDict[cate] = [i]
        else:
            CateDict[cate].append(i)
    return CateDict

# Function: get index for a specific period
def get_index_endpoints(index_dict, key_st, key_ed):
    '''
    Description
    -----------
        Get the start index and end index of the period.
    
    Parameters
    ----------
        index_dict: dict[month, index]
        key_st: float/int, start ym
        key_ed: float/int, end ym
    
    Output
    ------
        idx_st: float/int, start index
        idx_ed: float/int, end index
    '''
    idx_st = min(index_dict[key_st])
    idx_ed = max(index_dict[key_ed])
    return idx_st, idx_ed

def get_tn_dicts(Sample, date, id):
    # Time list
    Tlist = sorted(Sample[date].unique())
    T = len(Tlist)
    TDict = dict(zip(Tlist, range(T)))

    # Stock list
    Nlist = sorted(Sample[id].unique())
    N = len(Nlist)
    NDict = dict(zip(Nlist, range(N)))
    
    return TDict, NDict

# Function: convert the data to 3D array
def convert_array_old(Sample, date, id, cvars, retname):
    # Time list
    Tlist = sorted(Sample[date].unique())
    T = len(Tlist)
    TDict = dict(zip(Tlist, range(T)))

    # Stock list
    Nlist = sorted(Sample[id].unique())
    N = len(Nlist)
    NDict = dict(zip(Nlist, range(N)))

    # Characteristics
    K = len(cvars)

    # Convert to 3D array
    IMat = np.zeros([T, N])  
    RMat = np.zeros([T, N])
    CMat = np.zeros([T, N, K])
    for date, stkcd, exret, values in zip(Sample[date].values, Sample[id].values, Sample[retname].values, Sample[cvars].values):
        irow = TDict.get(date)
        icol = NDict.get(stkcd)
        IMat[irow, icol] = 1
        RMat[irow, icol] = exret
        CMat[irow, icol, :] = values

    # Return
    return TDict, NDict, IMat, RMat, CMat

def convert_array(df, vars_columns, index_columns=None, fill_value=0):
    
    '''
    Description
    -----------    
        Convert the input unbalanced panel data into a mask matrix and a multi-dimensional matrix.
    
    Parameters
    ----------
        df: pd.DataFrame, an unbalanced time-series data
        index_columns: list, the multi-index columns list, which usually be [time, stkcd].
                           If `index_columns` is specified, use the columns as the new index,
                           otherwise use the existing index.
        vars_columns: list, the value variables columns list.
        fill_value: float, optional, fill the missing value with fill_value
    
    Output
    ------
        index_dicts: a list of dictionary, the items in the list are multi-index dictionaries (position dictionaries).
        mask: np.array, the mask matrix, the number of dimensions is the same as the number of indexes
        array: np.array, the multi-dimensional array containing the information we want, such as the stock excess return, stock characteristics. 
        cvars_dict: dictionary, the dictionary of the information variables
    
    '''
    
    # Correct index columns (if just a string, convert it into a list)
    try:                                                                                                                                                                                                                                                                                               
        n_index = len(index_columns)                                                                                                                                                                                                                                                                     
    except TypeError:                                                                                                                                                                                                                                                                                  
        index_columns = [index_columns]                                                                                                                                                                                                                                                                            
        n_index = 1
    
    df = df[index_columns+vars_columns]
    # Set the columns as the new index
    if index_columns is not None:
        df = df.set_index(index_columns)

    # Extract the index and form a new MultiIndex of products
    index_levels = df.index.remove_unused_levels().levels
    index = pd.MultiIndex.from_product(index_levels)

    # Create an empty DataFrame with the new MultiIndex as the base
    index_df = pd.DataFrame([], index=index)

    # Merge the base with the DataFrame to get a balanced panel
    df_balanced = pd.merge(index_df, df, how='left', on=index.names)
    df_filled = df_balanced.fillna(fill_value)

    # Convert to arrays
    shape = tuple(len(l) for l in index_levels)
    array = np.squeeze(df_filled.to_numpy().reshape(shape + (-1,)))
    mask = df_balanced.notna().to_numpy().all(axis=1).astype(array.dtype).reshape(shape)
    
    # Index dictionaries
    index_dicts = []
    for i in range(n_index):
        dict_i = dict(zip(index_levels[i], range(len(index_levels[i]))))
        index_dicts.append(dict_i)
    cvars_dict = dict(zip(df_balanced.columns, range(len(df_balanced.columns))))
    
    del df, df_balanced, df_filled
    return index_dicts, mask, array, cvars_dict

def get_2d3d_metrics(df, vars_columns, index_columns=None, fill_value=0):
    
    '''
    Description
    -----------    
        Get all the dictionaries, 2D and 3D arrays we want.
    
    Parameters
    ----------
        df: pd.DataFrame, an unbalanced time-series data
        index_columns: list, the multi-index columns list, which usually be [time, stkcd].
                           If `index_columns` is specified, use the columns as the new index,
                           otherwise use the existing index.
        fill_value: float, optional, fill the missing value with fill_value
    
    Output
    ------
        TDict: dict[time, position], time position dictionary.
        NDict: dict[stock, position], stock position dictionary.
        IMat: np.array, 0-1 mask matrix.
        RMat: np.array, stock return matrix.
        CMat: np.array, stock characteristic matrix.
        
    '''    
    # Convert arrays
    Dicts, IMat, CMat2, Charas_Dict = convert_array(df, index_columns=index_columns, vars_columns=vars_columns, fill_value=0)
    # Seperate the array into the forms we want
    TDict, NDict, RMat, CMat = Dicts[0], Dicts[1], CMat2[:,:,0], CMat2[:,:,1:]

    del CMat2
    return TDict, NDict, IMat, RMat, CMat

#=================================================================================#
#=================================== Others =====================================#
#=================================================================================#
def torch_get_demean_weight(input_w):

    non_zero_mask = 1 * (input_w != 0)
    avg_daily_w = torch.sum(input_w, dim=1, keepdim=True) / torch.sum(non_zero_mask, dim=1, keepdim=True)
    demean_w = (input_w - avg_daily_w) * non_zero_mask
       
    return demean_w

def torch_get_qunt_weight(input_w, q):
    
    qunt = torch.quantile(input_w, q=q, dim=1)
    mask = 1 * ((input_w - qunt[0:1,:].T)<=0) | ((input_w - qunt[1:2,:].T)>=0)
    q_w = mask * input_w
    
    return q_w

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
    ## If the long/short < 0.8, then set it to 0.8
    lev_constr = torch.maximum(scl_pos/scl_neg, thres[0]) # [T,N,1]
    ## If the long/short >1.2, then set it to 1.2
    lev_constr = torch.minimum(lev_constr, thres[1]) # [T,N,1]
                                                                                                                                                                                                                                                                                                            
    # Transform 
    ## First we set the negative leverage to 1 every day
    Wt_neg = Wt_neg / scl_neg
    ## Then we set the positive leverage to 1 and then multiply it by the lev_contr,
    ## which is in [0.8, 1.2]
    Wt_pos = lev_constr * Wt_pos / scl_pos
    Wt = Wt_neg + Wt_pos                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
    return Wt 

def get_adv_weight(input_w, adv, q):  
    
    '''
    Description
    -----------
        This is to transform a stock signal matrix into an adv-weighted matrix, in a long-short form.
    
    Parameters
    ----------   
        input_w: Tensor, the input signal matrix, [T, N]
        adv: Tensor, the adv matrix, [T, N]
        q: list, the quntile you'd like to cut at, which is used to form long-short portfolio
            Example: [0.2, 0.8]

    Output
    ------
        adv: the adv-weighted weight matrix
    '''
    
    qunt = torch.quantile(input_w, q=q, dim=1)
    short_w = -1 * ((input_w - qunt[0:1,:].T)<=0)
    long_w = 1 * ((input_w - qunt[1:2,:].T)>=0)
    # Get a long-short -1,0,1 weight matrix
    mask_w = short_w + long_w 
    
    # transform input_w to 1 matrix
    input_w = torch.nan_to_num(input_w * torch.reciprocal(input_w))
    # Transform it to -1,0,1 matrix
    input_w = mask_w * input_w

    # Adv matrix
    adv_mat = adv/torch.sum(adv, dim=1, keepdim=True)
    adv_w = input_w * adv_mat

    return adv_w

def torch_get_ls_softmax_weight(input_w):
    
    # Get top and bottom mask
    short_mask = 1 * (input_w < 0)
    long_mask = 1 * (input_w > 0)
    
    # Get Long-side Portfolio
    long_w = input_w * long_mask
    exp_long = torch.exp(long_w) * long_mask
    long_w = exp_long/torch.sum(exp_long, dim=1, keepdim=True)
    
    # Get Short-side Portfolio
    short_w = input_w * short_mask
    exp_short = torch.exp(short_w) * short_mask
    short_w = exp_short/torch.sum(exp_short, dim=1, keepdim=True)        

    # The Ultimate Weight
    w = long_w - short_w

    return w

def torch_get_advthres_weight(input_w, adv, threshold=5):
    
    # ADV Threshold
    adv_thres = threshold * adv/torch.sum(adv, dim=1, keepdim=True)

    # Long Side
    long_w = torch.relu(input_w)
    w_adv_long = torch.min(long_w, adv_thres)
    
    # Short Side
    short_w = torch.relu(-input_w)
    w_adv_short = torch.min(short_w, adv_thres)

    w_adv = w_adv_long - w_adv_short

    return w_adv

def multi_merge(datasets, on, how):
    '''
    Description
    -----------  
        This function is designed the conduct merge operation in many datasets.
        
    Parameters
    ----------   
        datasets: tuple of pd.DataFrame(), (df1, df2, df3,...)
        on: string or list. The variables being merged on.
        how: string, the way of merging.
    
    Output
    ------
        merged_data: pd.DataFrame
    
    '''    
    for i, dataset in enumerate(datasets):
        if i==0:
            merged_data = dataset
        else:
            merged_data = pd.merge(merged_data, dataset, on=on, how=how)
    return merged_data

def get_st_ret_index(ndate, look_back, len_ret_batch):

    if look_back == 0:
        st_ret_index = torch.arange(0, ndate, len_ret_batch)
    else:
        st_ret_index = torch.arange(look_back - 1, ndate, len_ret_batch)
    last_ret_len = (ndate - 1) - st_ret_index[-1] + 1
    if (last_ret_len < len_ret_batch/2) & (len(st_ret_index)>1):
        st_ret_index = st_ret_index[:-1]

    return st_ret_index

# Function: winsorize
def winsorize(X, q):
    '''
    Description
    -----------  
        Winsorize the variable.
        Set value < quantile(q) to quantile(q)
        Set value > quantile(1-q) to quantile(1-q)
        
    Parameters
    ----------
        X: the variable
        q: winsorize threshold
        
    Output
    ------    
    Set value < quantile(q) to quantile(q)
    Set value > quantile(1-q) to quantile(1-q)
    '''
    q_l = np.nanquantile(X, q)
    q_u = np.nanquantile(X, 1 - q)
    X[X < q_l] = q_l
    X[X > q_u] = q_u
    return X

# Get Sharpe Ratio
def get_sr(ret, mul_freq=250):
    '''
    Description
    -----------  
        This is to get the Sharpe Ratio of the return.
    
    Parameters
    ---------- 
        ret: np.array
    
    Output
    ------
        sr = np.sqrt(250) * np.mean(ret) / np.std(ret)
       
    '''
    return np.sqrt(mul_freq) * np.mean(ret) / np.std(ret)

# Softmax
def softmax(x):
    '''
    Description
    -----------  
        This is the softmax function.
    
    Parameters
    ----------  
        x: np.array
    
    Output
    ------
        softmax = np.exp(x) / np.sum(np.exp(x))
       
    '''

    return np.exp(x) / np.sum(np.exp(x))

# Relu
def relu(x):
    return max(x,0)

def test_alpha(Y, Factor, is_rescale):
    # Create vector
    Alp = np.zeros([2, 3])

    # Adjust standard error
    if is_rescale:
        Y = Y / np.std(Y) * np.std(Factor['rmrf']) 

    # OLS regression 
    X_FF3 = sm.add_constant(Factor[['rmrf', 'smb', 'hml']]) 
    X_FF4 = sm.add_constant(Factor[['rmrf', 'smb', 'hml', 'mom']])
    X_FF5 = sm.add_constant(Factor[['rmrf', 'smb', 'hml', 'rmw', 'cma']])
    FF3 = sm.OLS(Y, X_FF3).fit()
    FF4 = sm.OLS(Y, X_FF4).fit()
    FF5 = sm.OLS(Y, X_FF5).fit()

    # Save results
    Alp[:, 0] = [FF3.params['const'], FF3.tvalues['const']]
    Alp[:, 1] = [FF4.params['const'], FF4.tvalues['const']]
    Alp[:, 2] = [FF5.params['const'], FF5.tvalues['const']]

    # Return
    return Alp  

def roll_SR(rt, col, nwin):
    rt['sr'] = np.sqrt(250) * rt[col].rolling(nwin).mean() / rt[col].rolling(nwin).std()
    return rt

#=================================================================================#
#================================ Post Analysis ==================================#
#=================================================================================#

'''
Description: This module contains functions for adjusting weights and evaluating the
             performance of the resulting portfolio.

Dependency: numpy, pandas

Date: 2021.3.07

Contributor: Keyu Zhou, Jialiang Lou
'''

# Get output dataframe of the stock weight
def output_stkwt_df(weight_mat, mask_mat, t_dict, n_dict, varname_w, varname_i):
    '''
    Description
    -----------    
        Use the weight matrix (numpy) to output a stock weight panel DataFrame.
    
    Parameters
    ----------
        weight_mat: numpy.array, the weight panel matrix, [T,N]
        ret_mat: numpy.array, the return panel matrix, [T,N]
        mask_mat: numpy.array, the mask panel matrix, [T,N]
        t_dict: dict[time, index], date dictionary.
        n_dict: dict[stkcd, index], stock stkcd dictionary.
        varname_w, varname_r, varname_i: string, the variable name of the stock weight, return, and mask columns.
        
    Output
    ------
        stkwt: pd.DataFrame, stock weight panel
        
    '''

    DataType = {'stkcd': 'int'}
    stkwt = pd.DataFrame(weight_mat, index=list(t_dict.keys()), columns=list(n_dict.keys())).reset_index()
    stkwt = pd.melt(stkwt, id_vars=['index'], var_name='stkcd',value_name=varname_w)
    stkwt = stkwt.rename(columns = {'index':'date'})
    
    mask = pd.DataFrame(mask_mat, index=list(t_dict.keys()), columns=list(n_dict.keys())).reset_index()
    mask = pd.melt(mask, id_vars=['index'], var_name='stkcd',value_name=varname_i)
    mask = mask.rename(columns = {'index':'date'})

    wt_df = pd.merge(stkwt, mask, on=['date', 'stkcd'], how='left')
    wt_df = wt_df.sort_values(by=['date', 'stkcd'], ascending=True).reset_index(drop=True)
    wt_df = wt_df.astype(DataType)
    
    wt_df = wt_df[wt_df[varname_i]==1][['date','stkcd',varname_w]]

    return wt_df


def get_adjusted_weight(Wt, interval=[0.8,1.2]):
    '''
    Description
    -----------  
        This is to adjust the stock weight.     
        Rule: sum of positive = 0.8-1.2
              sum of negative = 1
        The function is used under the numpy context.
    
    Parameters
    ----------  
        Wt: np.array, [1,N]
        interval: list, [floor, ceiling], the minimum and maximum leverage. 
    
    Output
    ------
        Wt: np.array, [1,N]
    
    '''
    # Ceil and Floor of the scale interval
    scl_ceil = interval[1]
    scl_floor = interval[0] 
    
    # Calculate the positive and negative total position in the raw weight.
    is_pos = (Wt > 0)
    is_neg = (Wt < 0)
    scl_pos = np.sum(Wt[is_pos]) 
    scl_neg = np.sum(-Wt[is_neg]) 
    
    # Transform the stock weights
    if scl_pos > 0 and scl_neg > 0:
        ## Standardize the negative position to 1
        Wt[is_neg] = Wt[is_neg] / scl_neg
        ## Standardize the positive position from 0.8 to 1.2
        if scl_pos / scl_neg > scl_ceil:
            Wt[is_pos] = scl_ceil * Wt[is_pos] / scl_pos 
        elif scl_pos / scl_neg < scl_floor:
            Wt[is_pos] = scl_floor * Wt[is_pos] / scl_pos 
        else:
            Wt[is_pos] = Wt[is_pos] / scl_neg
    # If either positive or negative position is zero:
    else:
        if scl_pos > 0:
            Wt[is_pos] = Wt[is_pos] / scl_pos
        if scl_neg > 0:
            Wt[is_neg] = Wt[is_neg] / scl_neg
    
    return Wt

# Function: get long-short portfolio weights
def get_longshort_weight(Wt, interval, how='equal'):
    '''
    Description
    -----------  
        This is to get the long short stock weight using the top and bottom signal stocks.     
        Rule: sum of positive = 1, only top X% stocks.
              sum of negative = 1, only bottom X% stocks.
        The function is used under the numpy context.
    
    Parameters
    ----------  
        Wt: np.array, [1,N]
        interval: list, [floor, ceiling], the bottom and top threshold for choosing stocks. 
        how: string, optional, 
                 which contains ['equal', 'softmax']
                 'equal' means the long-side and short-side is equal-weighted.
                 'softmax' means the long-side and short-side is transformed by softmax.
        
    Output
    ------
        Wt: np.array, [1,N]
    
    '''
    # Ceil and Floor of the scale interval
    scl_ceil = interval[1]
    scl_floor = interval[0]
    
    is_long = (Wt > np.nanquantile(Wt, scl_ceil))
    is_short = (Wt < np.nanquantile(Wt, scl_floor))
    is_zero = (Wt > np.nanquantile(Wt, scl_floor)) & (Wt < np.nanquantile(Wt, scl_ceil))
    
    if how == 'equal': 
        # Let the long and short side be equal-weighted
        if is_long.any():
            Wt[is_long] = 1 / np.sum(is_long)
        if is_short.any():
            Wt[is_short] = - 1 / np.sum(is_short)
        if is_zero.any():
            Wt[is_zero] = 0
    if how == 'softmax':
        # Let the long and short side be transformed by softmax functions.
        if is_long.any():
            Wt[is_long] = softmax(Wt[is_long])
        if is_short.any():
            Wt[is_short] = - softmax(Wt[is_short])
        if is_zero.any():
            Wt[is_zero] = 0
    
    return Wt

def get_yearmonth(df, time_col):
    '''
    Description
    -----------  
        This is to generate the 'year', 'month' and 'ym' variables
        Rule: ym = (year - 1960) * 12 + month -1
        The function is used under the pandas context.
    
    Parameters
    ----------
        df: pd.DataFrame()
        time_col: list, the name of the time variable
                  example: time_col = 'date'
    
    Output
    ------
        df: pd.DataFrame, the DataFrame with required time variables
    
    '''
    df = df.copy()
    df['year'] = df[time_col].map(lambda x: x.year)
    df['month'] = df[time_col].map(lambda x: x.month)
    df['ym'] = (df['year'] - 1960)*12 + df['month'] - 1
    return df

def df_get_size_adv_avg_rank(df, date_var, time_col, weight_methods, size_ranks=[5000], adv_ranks=[5000]):
        
    # Weights columns name
    weight_col = ['%s_weight'%method for method in weight_methods] 

    # Relative weight of each stock everyday
    rank_df = df[['date','stkcd','size','adv'] + weight_col]
    rank_df.set_index(['date','stkcd'], inplace=True)

    # Ranking Size and ADV every trading day
    rank_df['size'] = rank_df['size'].groupby(date_var).rank(ascending=False)
    rank_df['adv'] = rank_df['adv'].groupby(date_var).rank(ascending=False)

    size_avg_rank_all = pd.DataFrame()
    adv_avg_rank_all = pd.DataFrame()

    i = 0
    for size_rank, adv_rank in zip(size_ranks, adv_ranks):
        
        rank_size_df = rank_df[rank_df['size']<=size_rank]
        rank_size_df[weight_col] = abs(rank_size_df[weight_col])
        rank_size_df[weight_col] /= rank_size_df[weight_col].groupby(date_var).sum()
        rank_size_df[weight_col] = abs(rank_size_df[weight_col])
        rank_size_df[weight_col] /= rank_size_df[weight_col].groupby(date_var).sum()

        rank_adv_df = rank_df[rank_df['adv']<=adv_rank]
        rank_adv_df[weight_col] = abs(rank_adv_df[weight_col])
        rank_adv_df[weight_col] /= rank_adv_df[weight_col].groupby(date_var).sum()
        rank_adv_df[weight_col] = abs(rank_adv_df[weight_col])
        rank_adv_df[weight_col] /= rank_adv_df[weight_col].groupby(date_var).sum()

        # Average Size Rank TableW
        size_avg_rank = (rank_size_df[weight_col] * rank_size_df['size'].to_numpy().reshape(-1, 1)).groupby(date_var).sum().reset_index()
        size_avg_rank = get_yearmonth(df=size_avg_rank, time_col='date')

        size_avg_rank_tab = size_avg_rank.groupby(time_col)[weight_col].mean().reset_index()
        size_avg_rank_tab = size_avg_rank_tab.rename(columns={'%s_weight'%method: '%s_size_avg_rank_%d'%(method, size_rank) for method in weight_methods})

        size_avg_rank_whole = pd.DataFrame(size_avg_rank[weight_col].mean()).T
        size_avg_rank_whole = size_avg_rank_whole.rename(columns={'%s_weight'%method: '%s_size_avg_rank_%d'%(method, size_rank) for method in weight_methods})
        size_avg_rank_whole[time_col] = '%d-%d'%(size_avg_rank[time_col].min(),size_avg_rank[time_col].max())

        size_avg_rank_tab = size_avg_rank_tab.append(size_avg_rank_whole,ignore_index=True,sort=False)

        # Average ADV Rank Table
        adv_avg_rank = (rank_adv_df[weight_col] * rank_adv_df['adv'].to_numpy().reshape(-1, 1)).groupby(date_var).sum().reset_index()
        adv_avg_rank = get_yearmonth(df=adv_avg_rank, time_col='date')

        adv_avg_rank_tab = adv_avg_rank.groupby(time_col)[weight_col].mean().reset_index()
        adv_avg_rank_tab = adv_avg_rank_tab.rename(columns={'%s_weight'%method: '%s_adv_avg_rank_%d'%(method, adv_rank) for method in weight_methods})

        adv_avg_rank_whole = pd.DataFrame(adv_avg_rank[weight_col].mean()).T
        adv_avg_rank_whole = adv_avg_rank_whole.rename(columns={'%s_weight'%method: '%s_adv_avg_rank_%d'%(method, adv_rank) for method in weight_methods})
        adv_avg_rank_whole[time_col] = '%d-%d'%(adv_avg_rank[time_col].min(),adv_avg_rank[time_col].max())

        adv_avg_rank_tab = adv_avg_rank_tab.append(adv_avg_rank_whole,ignore_index=True,sort=False)

        if i == 0:
            size_avg_rank_all, adv_avg_rank_all = size_avg_rank_tab, adv_avg_rank_tab
        else:
            del size_avg_rank_tab['year'], adv_avg_rank_tab['year']
            # Concatenate
            size_avg_rank_all = pd.concat((size_avg_rank_all, size_avg_rank_tab), axis=1)
            adv_avg_rank_all = pd.concat((adv_avg_rank_all, adv_avg_rank_tab), axis=1)

        i += 1
        
    return size_avg_rank_all, adv_avg_rank_all


def df_get_ret(df, time_col, weight_methods, return_col):
    '''
    Description
    -----------  
        This function is used to get the return panel of the dataset, using the stock weight panel.
        
    Parameters
    ----------  
        df: pd.DataFrame(), the stock weight panel.
        time_col: the name of the time variable, which is used in the grouping calculation of the return.
        weight_methods: list, the methods of calculating weights. e.g: weight_methods = ['raw','adj','bal','ls_eq','ls_sfmx']
        return_col: the name of the return column
    
    Output
    ------
        ret_tab: pd.DataFrame, the return panel.
    
    '''

    for i,weight_method in enumerate(weight_methods):
        
        ret = df.groupby(time_col).apply(lambda X: np.sum(X['%s_weight'%weight_method]*X[return_col]))
        if i == 0:
            ret_tab = pd.DataFrame(ret).rename(columns={0:'%s_ret'%weight_method})
        else:
            ret_tab['%s_ret'%weight_method] = ret
    ret_tab = ret_tab.reset_index()
    return ret_tab

def df_get_sr(df, time_col, weight_methods):
    '''
    Description
    -----------  
        This function is used to get the Yearly Sharpr Ratio(SR) data of the dataset, using the return panel.
        
    Parameters
    ----------  
        df: pd.DataFrame(), the return panel.
        time_col: string, the name of the time variable, which is used in the grouping calculation of the return.
        weight_methods: list, the methods of calculating weights. e.g: weight_methods = ['raw','adj','bal','ls_eq','ls_sfmx']
    
    Output
    ------
        sr_tab: pd.DataFrame, the SR panel.
    
    '''
    return_col = ['%s_ret'%method for method in weight_methods] 
    sr_tab = df.groupby(time_col)[return_col].apply(lambda x: get_sr(x)).reset_index()
    sr_tab = sr_tab.rename(columns={'%s_ret'%method: '%s_sr'%method for method in weight_methods})

    sr_whole = pd.DataFrame(df[return_col].apply(lambda x: get_sr(x))).T
    sr_whole = sr_whole.rename(columns={'%s_ret'%method: '%s_sr'%method for method in weight_methods})
    sr_whole[time_col] = '%d-%d'%(df[time_col].min(),df[time_col].max())

    sr_tab = sr_tab.append(sr_whole,ignore_index=True,sort=False)
    return sr_tab

def df_get_std(df, time_col, weight_methods):
    '''
    Description
    -----------  
        This function is used to get the Daily Standard Deviation(SR) each year of the dataset, using the return panel.
        
    Parameters
    ----------  
        df: pd.DataFrame(), the return panel.
        time_col: string, the name of the time variable, which is used in the grouping calculation of the return.
        weight_methods: list, the methods of calculating weights. e.g: weight_methods = ['raw','adj','bal','ls_eq','ls_sfmx']
    
    Output
    ------
        std_tab: pd.DataFrame, the stdev panel.
    
    '''

    return_col = ['%s_ret'%method for method in weight_methods]
    std_tab = df.groupby(time_col)[return_col].apply(lambda x: np.std(x)).reset_index()
    std_tab = std_tab.rename(columns={'%s_ret'%method: '%s_std'%method for method in weight_methods})

    std_whole = pd.DataFrame(df[return_col].apply(lambda x: np.std(x))).T
    std_whole = std_whole.rename(columns={'%s_ret'%method: '%s_std'%method for method in weight_methods})
    std_whole[time_col] = '%d-%d'%(df[time_col].min(),df[time_col].max())

    std_tab = std_tab.append(std_whole,ignore_index=True,sort=False)
    return std_tab

def df_get_avgret(df, time_col, weight_methods):
    '''
    Description
    -----------  
        This function is used to get the average return data reported in yearly frequency.(Usually the monthly average return)
        The average return can be calculated in a pre-specified frequency
        
    Parameters
    ----------    
        df: pd.DataFrame(), the return panel (usually the daily return).
        time_col: string, the name of the time variable, which indicates the calculation frequency. (usually 'ym' to get the monthly return)
        weight_methods: list, the methods of calculating weights. e.g: weight_methods = ['raw','adj','bal','ls_eq','ls_sfmx']
    
    Output
    ------
        ret_m_tab: pd.DataFrame, the return panel.
    
    '''
    return_col = ['%s_ret'%method for method in weight_methods]
    ret_m = df.groupby(time_col)[return_col].apply(lambda X: np.prod(1 + X[return_col], axis=0) - 1).reset_index()
    ret_m = pd.merge(ret_m, df[['year',time_col]].drop_duplicates(), on='ym', how='left')
    ret_m = ret_m.groupby('year')[return_col].mean().reset_index()

    ret_m_whole = pd.DataFrame(ret_m[return_col].mean()).T
    ret_m_whole['year'] = '%d-%d'%(ret_m['year'].min(),ret_m['year'].max())

    ret_m_tab = ret_m.append(ret_m_whole,ignore_index=True,sort=False)
    ret_m_tab = ret_m_tab.rename(columns={'%s_ret'%method: '%s_ret_m'%method for method in weight_methods})

    return ret_m_tab

def df_get_turnover(df, time_col, weight_methods):
    '''
    Description
    -----------  
        This function is used to get the average daily turnover data reported in a required frequency.
        
    Parameters
    ----------  
        df: pd.DataFrame(), the stock weight panel.
        time_col: string, the report frequency of the turnover data.
        weight_methods: list, the methods of calculating weights. e.g: weight_methods = ['raw','adj','bal','ls_eq','ls_sfmx']
    
    Output
    ------
        pd.DataFrame(turnover, columns=cols): pd.DataFrame, the daily average turnover panel in a requiring frequency.
    
    '''    
    weight_col=['%s_weight'%method for method in weight_methods]
    turnover = []
    times = df[time_col].unique()
    time_list = times.tolist() + ['%d-%d'%(times.min(),times.max())]
    for time in time_list:
        if time == time_list[-1]:
            df_sub=df
        else:
            df_sub = df[df[time_col]==time]

        turnover_yr = []
        for weight_i in weight_col:
            weight_mat = pd.pivot_table(df_sub,values=weight_i,index='date',columns='stkcd', fill_value=0).values
            turnover_i =np.mean(np.sum(np.abs(np.diff(weight_mat,axis=0, prepend=0)), axis=1))
            turnover_yr.append(turnover_i)
        turnover.append([time]+turnover_yr)
    cols = [time_col] + ['%s_turnover'%method for method in weight_methods]
    return pd.DataFrame(turnover, columns=cols)


def get_trading_cost(weight_mat, cost_rate, t_axis=0):
    '''
    Description
    -----------  
        Calculate the trading cost return from a weight matrix.
    
    Parameters
    ----------   
        weight_mat: np.array, the stock weight matrix, usually be [T,N].
        cost_rate: float, The one-way trading cost rate.
        t_axis: int, the number of the time dimension.
    
    Output
    ------
        cost_ret: np.array, the trading cost in return
    
    '''
    # Get the change of weights
    diff_weight = np.concatenate([weight_mat[:1],np.diff(weight_mat, axis=t_axis)], axis=t_axis)
    # Calculate the trading cost
    cost_ret = cost_rate * np.sum(np.abs(diff_weight), axis=-1)
    return cost_ret

def df_get_trading_cost(df, time_col, cost_rate, weight_methods):
    '''
    Description
    -----------  
        Calculate the trading cost return in a DataFrame panel with multiple weight calculation methods.
    
    Parameters
    ----------   
        df: pd.DataFrame, the stock weight panel.
        time_col: string, the name of the time variable of which the frequency is used to report the trading cost
        cost_rate: float, the one-way trading cost rate.
        weight_methods: list, the ways of calculating weights, which are used as the prefixes of variables.
    
    Output
    ------
        cost_tab: pd.DataFrame, the trading cost panel.
    
    '''
    # Trading Cost
    cost_tab = pd.DataFrame()
    ## Set the time variable in the cost_tab
    cost_tab[time_col] = df[time_col].unique()
    ## Calculate the cost table
    for method in weight_methods:
        ### Transform the panel to 2-D array.
        weight_mat = pd.pivot_table(df,values='%s_weight'%method,index='date',columns='stkcd', fill_value=0).values
        ### Get the trading cost return.
        cost_ret = get_trading_cost(weight_mat=weight_mat, cost_rate=cost_rate)
        ### Append it to the cost_tab.
        cost_tab['%s_cost'%method] = cost_ret
    return cost_tab

def df_get_ac_ret(weight_df, ret_df, time_col, cost_rate, weight_methods):
    '''
    Description
    -----------  
        Get the after cost return table.
    
    Parameters
    ----------   
        weight_df: pd.DataFrame, the stock weight panel.
        ret_df: pd.DataFrame, the stock return dataframe.
        time_col: string, the name of the time variable of which the frequency is used to report the trading cost
        cost_rate: float, the one-way trading cost rate.
        weight_methods: list, the ways of calculating weights, which are used as the prefixes of variables.
    
    Output
    ------
        ac_ret_tab: pd.DataFrame, the after cost return table.
    
    '''
    
    # Set after-cost return table.
    ac_ret_tab = pd.DataFrame()
    ## after cost time variable name.
    ac_ret_tab[time_col] = ret_df[time_col]
    ## get the trading cost return table.
    cost_tab = df_get_trading_cost(df=weight_df, time_col='date', weight_methods=weight_methods, cost_rate=cost_rate)
    # Substract the trading cost return from the original return table.
    for method in weight_methods:
        ac_ret_tab['%s_ac_ret'%method] = ret_df['%s_ret'%method] - cost_tab['%s_cost'%method]
    
    return ac_ret_tab

def df_get_topw(df, tops, time_col, weight_methods):
    for i,top in enumerate(tops):
        for j,method in enumerate(weight_methods):
            print(i,j)
            weight_col = [method+'_weight']
            rank_wt = df[[time_col,'date'] + weight_col]
            rank_wt[['%s_rank'%method]] = df.groupby([time_col,'date'])[weight_col].rank(ascending=False)
            top_wt = rank_wt[rank_wt[method+'_rank']<=top]
            top_wt = pd.DataFrame(top_wt.groupby([time_col,'date'])[weight_col].sum())
            top_wt = pd.DataFrame(top_wt.groupby(time_col)[weight_col].mean()).reset_index()

            top_wt_whole = pd.DataFrame(top_wt[weight_col].mean()).T
            top_wt_whole[time_col] = '%d-%d'%(top_wt[time_col].min(),top_wt[time_col].max())
            top_wt_tab_i = top_wt.append(top_wt_whole,ignore_index=True,sort=False)
            top_wt_tab_i = top_wt_tab_i.rename(columns={'%s_weight'%method: '%s_top%s'%(method,top)})
            
            if i+j == 0:
                top_wt_tab = top_wt_tab_i
            else:
                top_wt_tab = top_wt_tab.merge(top_wt_tab_i, on=time_col, how='left')

    return top_wt_tab          
@njit
def std(array):
    'Calculate the standard deviation of an array (safe for singleton arrays)'
    if len(array) <= 1:
        return np.nan
    return np.sqrt(np.sum((array - np.mean(array))**2)/(len(array) - 1))

@njit
def mask_argsort(array, mask):
    'Return the indices that would sort an array, but only for the masked elements'
    array_temp = np.copy(array)
    array_temp[~mask] = np.NINF
    array_sort = np.argsort(array_temp)[np.sum(~mask):]
    return array_sort

@njit
def build_position(signal, limit_in, mask, init_cap):
    'Build the initial position based on the signal'
    # Sort (indirectly) by signal
    signal_sort = mask_argsort(signal, mask)

    # Initialize position and other variables
    position = np.zeros_like(signal)
    num = 0
    capital = 0

    # Build position
    # This process stops when the initial total capital is reached, or there is no stock available
    while len(signal_sort) > 0:
        # Find the stock with the best signal
        i = signal_sort[-1]
        signal_sort = signal_sort[:-1]

        # Use the trade limit as the position
        position[i] = limit_in[i]
        num += 1
        capital += limit_in[i]

        # Check whether reach the capital limit
        if capital >= init_cap:
            # Set the position to the capital limit
            position[i] -= capital - init_cap
            capital = init_cap
            break

    return position

@njit
def adjust_position(position, signal, limit_in, limit_out, limit_total, mask_buy, mask_sell, std_cutoff, sell_remain):
    'Adjust the daily position based on the signal and the previous position'
    # Signal cutoff based on the standard deviation
    signal_cutoff = std_cutoff * std(signal[mask_buy])

    # Sort (indirectly) by signal
    signal_sort_in = mask_argsort(signal, mask_buy)
    mask_position = (position > 0) & mask_sell
    signal_sort_out = mask_argsort(signal, mask_position)

    # Initialize position and other variables
    new_position = position.copy()
    capital_in = 0
    capital_out = 0

    # Adjust position
    # This process stops when the signal difference is below the cutoff, or there is no stock available
    while len(signal_sort_out) > 0 and len(signal_sort_in) > 0 \
        and signal[signal_sort_in[-1]] - signal[signal_sort_out[0]] > signal_cutoff:

        # Find the stock to trade and calculate capital limits
        # If the capital is not zero, continue trading the previous stock
        if capital_in == 0:
            stock_in = signal_sort_in[-1]
            capital_in = np.max(np.array([np.min(np.array([limit_in[stock_in], limit_total[stock_in] - position[stock_in]])), 0]))
        if capital_out == 0:
            stock_out = signal_sort_out[0]
            # If we can sell all the stock's position
            if position[stock_out] < limit_out[stock_out] * (1 + sell_remain):
                capital_out = position[stock_out]
            else:
                capital_out = limit_out[stock_out]

        # Adjust position
        capital_trade = np.min(np.array([capital_in, capital_out]))
        new_position[stock_in] += capital_trade
        new_position[stock_out] -= capital_trade
        capital_in -= capital_trade
        capital_out -= capital_trade

        # Remove the traded stock from the list
        # If the capital is not zero, the stock is still in the list for the next trade
        if capital_in == 0:
            signal_sort_in = signal_sort_in[:-1]
        if capital_out == 0:
            signal_sort_out = signal_sort_out[1:]

    return new_position

@njit
def truncate_weight(wt, level):
    '''
    Truncate an array such that each element is not greater than `level` of the array

    Only make sense for an array of all elements with the same sign.
    '''
    # Check feasibility
    if len(wt) < 1 / level:
        # Now the best solution is to equally distribute among each element
        level = 1 / len(wt)
        return np.full_like(wt, np.mean(wt))

    # Standardize such that the weights sum to 1 (also remove the sign)
    sum_weight = np.sum(wt)
    result_wt = wt / sum_weight

    # Truncate the values that are greater than `level`, equally distribute to the others
    # Do this recursively until no element is greater than `level`
    already_truncated_idx = []
    truncate_idx = np.where(result_wt > level)[0]
    while len(truncate_idx) > 0:
        non_truncate_idx = np.array(list(set(range(len(result_wt))) - set(truncate_idx)
                                         - set(already_truncated_idx)))
        truncated_weights = np.sum(result_wt[truncate_idx] - level)
        result_wt[truncate_idx] = level
        result_wt[non_truncate_idx] += truncated_weights / len(non_truncate_idx)
        already_truncated_idx.extend(truncate_idx)
        truncate_idx = np.where(result_wt > level)[0]

    # Don't forget to multiply the original sum
    return result_wt * sum_weight
    
def convert_array(df, index_columns=None, fill_value=0):
    '''
    Convert a DataFrame with MultiIndex to a multi-dimensional array.
    In other words, convert an unbalanced panel to a balanced panel.

    If `index_columns` is specified, use the columns as the new index,
    otherwise use the existing index.

    Use `fill_value` to fill in all missing values, including those in
    the original DataFrame.
    '''
    # Set the columns as the new index
    if index_columns is not None:
        df = df.set_index(index_columns)

    # Extract the index and form a new MultiIndex of products
    index_levels = df.index.remove_unused_levels().levels
    index = pd.MultiIndex.from_product(index_levels)

    # Create an empty DataFrame with the new MultiIndex as the base
    index_df = pd.DataFrame([], index=index)

    # Merge the base with the DataFrame to get a balanced panel
    df_balanced = pd.merge(index_df, df, how='left', on=index.names)
    df_filled = df_balanced.fillna(fill_value)
    print(df_filled)
    # Convert to arrays
    shape = tuple(len(l) for l in index_levels)
    array = df_filled.to_numpy().reshape(shape + (-1,))
    mask = df_balanced.notna().to_numpy().any(axis=1).reshape(shape)

    return array, mask, index_levels, df_balanced.columns

def backtest(signal, ret, trade_limit, hold_limit, mask_buy, mask_sell, init_cap, shorting=False, market=None,
             rebalance=float('inf'), cost_rate=0, std_cutoff=0, rescale=None, sell_remain=0.25):
    '''
    Backtest a signal output of a model

    This will form a position series based on the signal. The initial position is formed by stocks with the best initial signal.
    It will be adjusted by trading stocks with the worst signal for stocks with the best signal.

    Parameters
    ----------
    signal : array-like
        The signal output of the model.
    ret : array-like
        The return series.
    trade_limit : array-like or tuple[array-like, array-like]
        The maximum dollar amount to trade for each stock on each day. If a binary tuple is given, the first element is the
        buy limit and the     second element is the sell limit.
    hold_limit : array-like
        The maximum position for each stock on each day.
    mask_buy : array-like[bool]
        The mask for the stocks to buy. The mask is `True` for the stocks available to buy.
    mask_sell : array-like[bool]
        The mask for the stocks to sell. The mask is `True` for the stocks available to sell.
    init_cap : float
        The initial total capital. If allow shorting, the initial capital will be the same for both long and short.
    shorting : bool, optional
        Whether to allow shorting. If `True`, both long and short initial positions will be formed to `init_cap`.
        Default is `False`.
    market : array-like, optional
        The market return series. This is used to hedge the market in the long-short portfolio. Must be provided if `shorting`
        is `True`. Will be ignored if `shorting` is `False`.
    rebalance : float, optional
        The ratio of the long position to the short position to trigger rebalancing. Will be ignored if `shorting` is `False`.
        Default is `inf` (no rebalancing).
    cost_rate : float, optional
        The one-way trading cost rate (calculated only once during position adjustment).
    std_cutoff : float, optional
        The standard deviation cutoff of the signal during position adjustment. If the signal difference is below this cutoff,
        the adjustment will stop.
    rescale : tuple[float, float], optional
        The tuple of `(target, threshold)` for rescaling. The rescaling will be performed when the total position exceeds
        `threshold` times the initial position, to `target` times the initial position. If `None`, no rescaling.
    sell_remain : float, optional
        Sell all of the stock when the remaining position is less than this percentage of `trade_limit` after selling.

    Returns
    -------
    turnover : np.ndarray
        The resulting turnover series.
    ret_series : np.ndarray
        The resulting return series.
    position : np.ndarray
        The resulting position series.

    Raises
    ------
    ValueError
        - If `shorting` is `True` but `market` is not provided.
        - If `shorting` is `True` and `init_cap` is too large for long and short positions to be disjoint.
    '''
    # Adjust parameters
    try:
        limit_in, limit_out = trade_limit
        assert limit_in.shape == limit_out.shape == signal.shape
    except (ValueError, AssertionError):
        limit_in = limit_out = trade_limit

    # Daily return adjustment
    return_adjust = np.nan_to_num(ret)
    return_adjust += 1

    # List for keeping results
    position = []
    turnover = []
    ret_series = []

    # Build position
    position_day = build_position(signal[0], limit_in[0], mask_buy[0], init_cap)
    position_day *= 1 - cost_rate
    if shorting:
        if market is None:
            raise ValueError('Market return must be provided if allow shorting')
        position_day_short = build_position(-signal[0], limit_out[0], mask_sell[0], init_cap)
        position_day_short *= 1 - cost_rate
        # Check if the initial positions are disjoint
        if np.any((position_day != 0) & (position_day_short != 0)):
            raise ValueError('The initial capital is too large for long and short positions to be disjoint')
        position_day -= position_day_short
    capital = np.sum(abs(position_day))
    position.append(position_day)

    # Adjust position (daily)
    for i in range(1, signal.shape[0]):
        # Position change due to returns
        position_last = position_day
        position_day = position_day * return_adjust[i-1]

        # Adjust daily position
        if shorting:
            position_day_long = position_day.copy()
            position_day_long[position_day_long < 0] = 0
            position_day_short = -position_day.copy()
            position_day_short[position_day_short < 0] = 0

            position_new_long = adjust_position(position_day_long, signal[i], limit_in[i], limit_out[i], hold_limit[i], mask_buy[i], mask_sell[i], std_cutoff, sell_remain)
            position_new_short = adjust_position(position_day_short, -signal[i], limit_out[i], limit_in[i], hold_limit[i], mask_sell[i], mask_buy[i], std_cutoff, sell_remain)
            position_new = position_new_long - position_new_short
        else:
            position_new = adjust_position(position_day, signal[i], limit_in[i], limit_out[i], hold_limit[i], mask_buy[i], mask_sell[i], std_cutoff, sell_remain)

        # Calculate trading cost, and distribute cost among `stock_in`
        position_change = position_new - position_day
        position_change[position_change < 0] = 0
        trading_cost_adj = 0
        if shorting:
            trading_cost_sum = np.sum(position_change * cost_rate * 2)
            position_change_long = position_new_long - position_day_long
            position_change_long[position_change_long < 0] = 0
            if (change_sum := np.sum(position_change_long)) > 0:
                trading_cost = position_change_long / change_sum * trading_cost_sum
            else:
                trading_cost = np.zeros_like(position_change_long)
                trading_cost_adj = trading_cost_sum
            position_new_long -= trading_cost
        else:
            trading_cost = position_change * cost_rate * 2
        position_new -= trading_cost

        # Calculate turnover and return
        turnover_day = np.sum(abs(position_new - position_last)) / capital
        ret_day = (np.sum(position_new - position_last) - trading_cost_adj) / capital
        if shorting:
            # Hedge the market
            ret_day -= np.sum(position_last) / capital * market[i]

        # Rebalance long and short positions
        if shorting and ((ls_ratio := np.sum(position_new_long) / np.sum(position_new_short)) >= rebalance or ls_ratio <= 1 / rebalance):
            position_new_long *= (ls_ratio + 1) / (2 * ls_ratio)
            position_new_short *= (ls_ratio + 1) / 2
            position_new = position_new_long - position_new_short
            ret_day -= abs(ls_ratio - 1) / (ls_ratio + 1) * cost_rate  # Trading cost for rebalancing

        # Rescale position
        capital = np.sum(abs(position_new))
        if rescale is not None and (cap_ratio := capital / init_cap) > rescale[1]:
            position_new *= rescale[0] / cap_ratio
            capital = rescale[0] * init_cap

        # Append results
        position_day = position_new
        position.append(position_day)
        turnover.append(turnover_day)
        ret_series.append(ret_day)

    turnover.append(np.nan)
    ret_series.append(np.nan)
    return np.array(ret_series), np.array(turnover), np.stack(position)

def set_number_format(sheet, cell_range, format):
    '''
    Set the number format of a range of cells in an `openpyxl` worksheet.

    Parameters
    ----------
    sheet : Worksheet
        The worksheet.
    cell_range : tuple[tuple[int, int], tuple[int, int]]
        The start and end coordinates of the cells `((start_row, start_col),
        (end_row, end_col))`. Note that both row and column indices start from
        1, and the end coordinate is exclusive.
    format : str
        The number format. See `openpyxl.styles.numbers.BUILTIN_FORMATS` for
        available formats.
    '''
    start_row, start_col = cell_range[0]
    end_row, end_col = cell_range[1]
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            sheet.cell(row, col).number_format = format
