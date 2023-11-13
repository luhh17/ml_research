import pandas as pd
import numpy as np
import torch

'''
根据股票的第一个观测和最后一个观测来填补面板
'''
def expand_balanced_panel(df, date_var='mthcaldt', id_var='permno'):
    def fill_missing(_df, _date_list):
        sub_date = np.sort(_df[date_var].unique())
        should_date = _date_list[np.where((_date_list >= sub_date[0]) & (_date_list <= sub_date[-1]))]
        if should_date.shape[0] != sub_date.shape[0]:
            _df = pd.merge(pd.DataFrame({date_var: should_date}), _df, how='left', on=date_var)
            _df[id_var] = _df[id_var].fillna(method='ffill')
        return _df
    all_trd_dates = np.sort(df[date_var].unique())
    expand_df = df.groupby(id_var).apply(lambda x: fill_missing(x, all_trd_dates))
    expand_df.index = expand_df.index.droplevel()
    return expand_df


'''
构建股票代码和时间矩阵
'''
def convert_array(df, ret_var,index_columns=None, fill_value=0):
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
    # print(df_filled)
    # Convert to arrays
    shape = tuple(len(l) for l in index_levels)
    array = df_filled.to_numpy().reshape(shape + (-1,))

    # the mask is used to indicate whether the ret variable is missing or not
    mask = df_balanced[ret_var].notna().to_numpy().reshape(shape)
    df_filled = df_filled.reset_index()
    df_filled['date'] = df_filled['date'].apply(lambda x: x.year * 10000 + x.month * 100 + x.day).astype('int32')
    date = df_filled['date'].to_numpy().reshape(shape + (-1,))
    stkcd = df_filled['stkcd'].drop_duplicates().to_numpy().astype('int32')
    '''
    mask N * T
    stkcd N
    '''
    return array, mask, index_levels, df_balanced.columns, date, stkcd




