'''
Description: This module is to define those functions we would like to use when preprocessing or evaluating the data.
             Most of the functions are under the numpy and pandas context.

Dependency:  numpy, tensorflow

Version:  2.0 or newer

Date: 2022.03.05

Contributor: Jialiang Lou, Keyu Zhou
'''
import os
import numpy as np
import pandas as pd
import torch_package.functions as functions

#=================================================================================#
#================================ Import Data ==================================#
#=================================================================================#

# Variable names
date_var = 'date'
stock_var = 'stkcd'
mkt_var = 'mktret'
ret_var = 'ret_open'

# Parameters
capital_tot = 1e8
adv_limit = 0.01
adv_limit_hold = 0.05
std_cutoff = 1

def post_analysis(data_path, in_path, wt_name, date_var, stock_var, mkt_var, capital_tot, adv_limit, adv_limit_hold, std_cutoff, trading_cost, weight_methods, size_ranks, adv_ranks, **kwargs):

    # Import Data
    stkwt = pd.read_stata(in_path + '/' + wt_name)
    stkwt[date_var] = stkwt[date_var].astype('datetime64')
    #stkwt.rename(columns={'permno':'stkcd'}, inplace=True)

    # Merge Returns
    ret_df = pd.read_pickle(os.path.join(data_path +'/'+'Ashare_Stock_return.pkl'))
    ret_df[stock_var] = ret_df[stock_var].astype(int)
    stkwt = pd.merge(stkwt, ret_df, on=[date_var, stock_var], how='left')

    # Merge Market Data into it
    mkt_df = pd.read_pickle(data_path+'/'+'CSI_500.pkl')
    stkwt = pd.merge(stkwt, mkt_df, on=date_var, how='left')
    stkwt['exret'] = stkwt[ret_var] - stkwt[mkt_var]

    # ADV Result Transformation
    ### Get market return series
    mkt_df.set_index(date_var, inplace=True)
    date_series = stkwt[date_var].unique()
    mkt_df_subset = mkt_df.loc[date_series]
    market = mkt_df_subset[mkt_var].to_numpy()

    ### Trading ADV
    trading_adv_file = 'trading_adv_info.pkl'
    trading_adv_df = pd.read_pickle(os.path.join(data_path, trading_adv_file))
    trading_adv_df[stock_var] = trading_adv_df[stock_var].astype(int)
    stkwt = pd.merge(stkwt, trading_adv_df, on=[stock_var,date_var], how='left')
    stkwt.set_index([date_var, stock_var], inplace=True)

    ### Convert to arrays
    array, mask, _, columns = functions.convert_array(stkwt)
    signal = array[..., columns.get_loc('weight')].astype(float)
    ret = array[..., columns.get_loc(ret_var)].astype(float)
    stock_adv = array[..., columns.get_loc('adv')].astype(float)
    mask_buy = array[..., columns.get_loc('mask_buy')].astype(bool) * mask
    mask_sell = array[..., columns.get_loc('mask_sell')].astype(bool) * mask
    stock_adv_limit = adv_limit * stock_adv
    stock_adv_limit_hold = adv_limit_hold * stock_adv

    ### Backtest with ADV trading strategy
    ret_bc, turnover_bc, position_bc = functions.backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy, mask_sell, capital_tot,
                                                          shorting=True, market=market, rebalance=1, cost_rate=0, std_cutoff=std_cutoff)
    ret_ac, turnover_ac, position_ac = functions.backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy, mask_sell, capital_tot,
                                                          shorting=True, market=market, rebalance=1, cost_rate=trading_cost, std_cutoff=std_cutoff)

    ### ADV Ret Table
    adv_ret = pd.DataFrame(ret_bc).rename(columns={0:'adv_bc_ret'})
    adv_ret['adv_ac_ret'] = ret_ac
    adv_ret['date'] = date_series
    adv_ret = adv_ret.fillna(0)

    ### ADV TO Table
    adv_to = pd.DataFrame(turnover_bc).rename(columns={0:'adv_bc_turnover'})
    adv_to['adv_ac_turnover'] = turnover_ac
    adv_to['date'] = date_series
    adv_to['year'] = adv_to['date'].dt.year
    adv_to = adv_to.fillna(0)
    adv_to_tab = adv_to.groupby('year')[['adv_bc_turnover','adv_ac_turnover']].mean().reset_index()
    adv_to_tab_whole = pd.DataFrame(adv_to_tab[['adv_bc_turnover','adv_ac_turnover']].mean()).T
    adv_to_tab_whole['year'] = '%d-%d'%(adv_to_tab['year'].min(),adv_to_tab['year'].max())
    adv_to_tab = adv_to_tab.append(adv_to_tab_whole)

    # Get Weights Panel
    stkwt = stkwt.reset_index()
    stkwt['raw_weight'] = stkwt['weight']
    stkwt['adj_weight'] = stkwt.groupby('date')['weight'].apply(lambda X: functions.get_adjusted_weight(X, [0.8, 1.2]))
    stkwt['bal_weight'] = stkwt.groupby('date')['weight'].apply(lambda X: functions.get_adjusted_weight(X, [1, 1]))
    stkwt['ls_eq_weight'] = stkwt.groupby('date')['weight'].apply(lambda X: functions.get_longshort_weight(X, [0.1, 0.9], how='equal'))
    stkwt['ls_sfmx_weight'] = stkwt.groupby('date')['weight'].apply(lambda X: functions.get_longshort_weight(X, [0.1, 0.9], how='softmax'))
    stkwt['long_weight'] = np.maximum(stkwt['bal_weight'], 0)
    stkwt['short_weight'] = np.minimum(stkwt['bal_weight'], 0)
    stkwt = functions.get_yearmonth(df=stkwt, time_col='date')

    # Size and ADV Average Rank Table
    size_avg_rank_tab, adv_avg_rank_tab = functions.df_get_size_adv_avg_rank(df=stkwt, date_var=date_var, time_col='year', weight_methods=weight_methods, size_ranks=size_ranks, adv_ranks=adv_ranks)

    # Before Cost Analysis
    ## Get Return Table
    ret_tab = functions.df_get_ret(df=stkwt, time_col='date', weight_methods=weight_methods, return_col='exret')
    ret_tab = functions.get_yearmonth(df=ret_tab, time_col='date')
    ret_tab = pd.merge(ret_tab, adv_ret, on='date', how='left')

    ## Get Sharpe Ratio Table
    sr_tab = functions.df_get_sr(df=ret_tab, time_col='year', weight_methods=weight_methods + ['adv_bc', 'adv_ac'])

    ## Get Stdard Deviation Table
    std_tab = functions.df_get_std(df=ret_tab, time_col='year', weight_methods=weight_methods + ['adv_bc', 'adv_ac'])

    ## Get Monthly Average Return Table
    ret_m_tab = functions.df_get_avgret(df=ret_tab, time_col='ym', weight_methods=weight_methods + ['adv_bc', 'adv_ac'])

    ## Get Turnover Table
    turnover_tab = functions.df_get_turnover(df=stkwt, time_col='year', weight_methods=weight_methods)
    turnover_tab = pd.merge(turnover_tab, adv_to_tab, on='year', how='left')

    ## Get Top Stock Table
    top_tab = functions.df_get_topw(df=stkwt, tops=[1, 10], time_col='year', weight_methods=['long'])

    # Merge all Table and Form the Result Table
    result_tab = functions.multi_merge((sr_tab, ret_m_tab, std_tab, turnover_tab, top_tab), on='year', how='left')
    result_tab = result_tab.set_index('year').T

    # After Cost Analysis
    ## Get the Aftercost Return Table
    ac_ret_tab = functions.df_get_ac_ret(weight_df=stkwt, ret_df=ret_tab, time_col='date', cost_rate=0.0020, weight_methods=weight_methods)
    ac_ret_tab = functions.get_yearmonth(df=ac_ret_tab, time_col='date')

    ## Get the Aftercost Sharpe Ratio Table
    ac_weight_methods = ['%s_ac'%method for method in weight_methods]
    ac_sr_tab = functions.df_get_sr(df=ac_ret_tab, time_col='year', weight_methods=ac_weight_methods)

    ## Get the Aftercost Standard Deviation Table
    ac_std_tab = functions.df_get_std(df=ac_ret_tab, time_col='year', weight_methods=ac_weight_methods)

    ## Get the Aftercost Monthly Average Return Table
    ac_ret_m_tab = functions.df_get_avgret(df=ac_ret_tab, time_col='ym', weight_methods=ac_weight_methods)

    ## Merge all Table and Form  the Aftercost Result Table
    ac_result_tab = functions.multi_merge((ac_sr_tab, ac_ret_m_tab, ac_std_tab, size_avg_rank_tab, adv_avg_rank_tab), on='year', how='left')
    ac_result_tab = ac_result_tab.set_index('year').T

    # Append into the same table
    result_tab = result_tab.append(ac_result_tab)

    return stkwt, ret_tab, ac_ret_tab, result_tab


