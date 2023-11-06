'''
Post Analysis
--------
Functions for backtesting a signal output of a model.

Functions
---------
- `balance_weight`: calculate the balanced weights
- `calculate_turnover`: calculate turnover
- `calculate_port_ret`: calculate portfolio returns
- `calculate_sr`: calculate Sharpe ratio
- `backtest`: backtest a signal output of a model with a trading strategy
- `set_number_format`: set the number format of cells in an Excel worksheet

Author: Keyu Zhou
Revised by yetian 2023-11-03, add max_drawdown,change function calculate_port_ret to calculate_port_ret_c2c
'''

import numpy as np
import pandas as pd
from numba import njit
import pdb

@njit
def std(array):
    'Calculate the standard deviation of an array (safe for singleton arrays)'
    if len(array) <= 1:
        return np.nan
    return np.sqrt(np.sum((array - np.mean(array))**2)/(len(array) - 1))

@njit
def mask_argsort(array, mask):
    '''
    Default: Asscending
    '''
    'Return the indices,  only for the masked elements'
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

    # Convert to arrays
    shape = tuple(len(l) for l in index_levels)
    array = df_filled.to_numpy().reshape(shape + (-1,))
    mask = df_balanced.notna().to_numpy().any(axis=1).reshape(shape)

    return array, mask, index_levels, df_balanced.columns

def balance_weight(df, weight_column, id_columns=None, truncate_level=1, min_sum=1e-6):
    '''
    Calculate the balanced weights using a DataFrame of weights with panel data

    The weights are balanced such that both long-leg (positive elements) and short-leg
    (negative elements) will sum to 1.

    Specify the identity columns (e.g. date, stock id) in `id_columns`, otherwise use the
    index of the DataFrame as identity columns.

    Specify `truncate_level` to truncate weights at this level (see `truncate_weight`).

    The weights that sum less than `min_sum` in absolute value in the cross-section will
    be set to 0.
    '''
    # Set the columns as the new index
    if id_columns is not None:
        df = df.set_index(id_columns)

    # Select long-leg and short-leg weights
    wt_long = df.loc[df[weight_column] > 0, weight_column]
    wt_short = df.loc[df[weight_column] < 0, weight_column]

    # Standardize the weights in each cross-section such that sum to 1
    wt_long_sum = wt_long.groupby(level=0).sum()
    wt_short_sum = -wt_short.groupby(level=0).sum()
    wt_long /= wt_long_sum
    wt_short /= wt_short_sum

    # Truncate weights
    if truncate_level < 1:
        wt_long = wt_long.groupby(level=0).transform(lambda x: truncate_weight(x.values, level=truncate_level))
        wt_short = wt_short.groupby(level=0).transform(lambda x: truncate_weight(x.values, level=truncate_level))

    # Set weights that sum less than `min_sum` to 0
    wt_long.loc[wt_long_sum.index[wt_long_sum < min_sum]] = 0
    wt_short.loc[wt_short_sum.index[wt_short_sum < min_sum]] = 0

    # Concatenate data
    wt = pd.concat([wt_long, wt_short], axis=0)
    wt = wt.reindex(df.index, fill_value=0)

    return wt

def calculate_turnover(df, id_columns=None):
    '''
    Calculate turnover using a DataFrame of weights with panel data

    Specify the identity columns (e.g. date, stock id) in `id_columns`, otherwise use the
    index of the DataFrame as identity columns.
    '''
    # Convert the DataFrame to a balanced multi-dimensional array
    bal_arr, _, (date_idx, *_), columns = convert_array(df, id_columns)

    # Calculate turnover
    turnover = abs(np.diff(bal_arr, axis=0, prepend=np.nan)).sum(axis=1)
    bc_df = pd.DataFrame(turnover, index=date_idx, columns=columns)
    ac_df = pd.DataFrame(turnover, index=date_idx, columns=['ac_' + c for c in columns])
    res_df = pd.concat([ac_df, bc_df], axis=1)
    return res_df

def calculate_port_ret(df, id_columns=None):
    '''
    Calculate portfolio return series using a DataFrame of weights and returns

    Specify the identity columns (e.g. date, stock id) in `id_columns`, otherwise use the
    index of the DataFrame as identity columns.
    '''
    # Convert the DataFrame to a balanced multi-dimensional array
    bal_arr, _, (date_idx, *_), columns = convert_array(df, id_columns)
    # Separate weights and returns
    ret_c2o_idx = columns.get_loc('ret_close_to_open')
    ret_o2c_idx = columns.get_loc('ret_open_to_close')
    ret_mkt_idx = columns.get_loc('mktret')

    columns = columns.delete([ret_c2o_idx,ret_o2c_idx,ret_mkt_idx])
    wt_arr = np.delete(bal_arr, [ret_c2o_idx,ret_o2c_idx,ret_mkt_idx], axis=-1)
    # get the portfolio weights the day before, used as the close-to-open weights
    wt_arr_before=wt_arr.copy()
    wt_arr_before[1:,:,:]=wt_arr[:-1,:,:]
    wt_arr_before[0,:,:]=0

    ret_c2o_arr = bal_arr[..., [ret_c2o_idx]]
    ret_o2c_arr = bal_arr[..., [ret_o2c_idx]]
    ret_mkt_arr = bal_arr[..., [ret_mkt_idx]]
    
    # the close-to-close return is composed of two parts, the close-to-open return and the open-to-close return
    # the first's weight is determined by wt_arr_before, the second's weight is determined by wt_arr
    port_ret_c2c = ((wt_arr_before * ret_c2o_arr)+(wt_arr * ret_o2c_arr)).sum(axis=1)
    port_exret_c2c= ((wt_arr_before * ret_c2o_arr)+(wt_arr * (ret_o2c_arr-ret_mkt_arr))).sum(axis=1)

    # Calculate portfolio return series
    trading_cost = np.abs(np.concatenate([wt_arr[:1], np.diff(wt_arr, axis=0)], axis=0) * 0.0015)
    trading_cost = np.sum(trading_cost, axis=1)
    
    ac_ret = port_ret_c2c - trading_cost
    ac_df = pd.DataFrame(ac_ret, index=date_idx, columns=['ac_' + c for c in columns])
    bc_df = pd.DataFrame(port_ret_c2c, index=date_idx, columns=columns)
    res_df = pd.concat([ac_df, bc_df], axis=1)

    ac_exret = port_exret_c2c - trading_cost
    ac_df = pd.DataFrame(ac_exret, index=date_idx, columns=['ac_' + c for c in columns])
    bc_df = pd.DataFrame(port_exret_c2c, index=date_idx, columns=columns)
    exres_df = pd.concat([ac_df, bc_df], axis=1)

    return res_df,exres_df

def max_drawdown(x,ratio=False):
    y = np.cumsum(x)
    j = np.argmax(np.maximum.accumulate(y)-y)
    if j == 0:
        i = np.nan
        j = np.nan
        dd = np.nan
    else:
        i = np.argmax(y[:j])
        if ratio:
            dd = (y[i]-y[j])/y[i]
        else:
            dd = y[i]-y[j]

    return dd,x.index[i].date().strftime('%Y-%m-%d'),x.index[j].date().strftime('%Y-%m-%d')

def calculate_sr(ret_series, mul_freq=250):
    '''
    Calculate mean, standard deviation, Sharpe ratio and max drawdown using return series

    Use `mul_freq` to annualize Sharpe ratio according to your data frequency, i.e. how
    many observations each year (250 for daily, 50 for weekly and 12 for monthly).
    '''
    mean = ret_series.mean(axis=0)
    std = ret_series.std(axis=0)
    sr = mean / std * np.sqrt(mul_freq)
    
    drawdown = ret_series.apply(max_drawdown, axis=0).T
    

    sum_df=pd.concat([mean, std, sr,drawdown], axis=1).T
    sum_df.index=['mean', 'std', 'SR','max_dd','dd_start','dd_end']

    return sum_df

def backtest(signal, ret, trade_limit, hold_limit, mask_buy, mask_sell, init_cap, shorting=False, market=None,
             rebalance=float('inf'), cost_rate=0, std_cutoff=0, rescale=None, sell_remain=0.25):
    '''
    Backtest a signal output of a model

    This will form a position series based on the signal. The initial position is formed by stocks with the best initial signal.
    It will be adjusted by trading stocks with the worst signal for stocks with the best signal.

    Parameters
    ----------
    signal : array-like, shape(T, n_stocks)
        The signal output of the model.
    ret : array-like
        The return series.
    trade_limit : array-like or tuple[array-like, array-like]
        The maximum dollar amount to trade for each stock on each day. If a binary tuple is given, the first element is the
        buy limit and the second element is the sell limit.
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

    # Adjust position
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
