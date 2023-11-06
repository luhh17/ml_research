import pandas as pd
import numpy as np


def sorting_portfolio(df, group_by_var, group_num=10, weight_var=None, return_var='ret', date_var='date', id_var='stkcd'):
    '''
    计算分组投资组合的收益率
    输入：df为面板数据
    group_by_var为分组变量
    group_num为分组数
    weight_var为权重变量, 默认为空即等权
    '''
    def cal_group_ret(expand_monthly, weight_var):
        def wavg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            try:
                return (d * w).sum() / w.sum()
            except ZeroDivisionError:
                return np.nan

        port = expand_monthly.groupby([date_var, f'{group_by_var}_group']).apply(
            lambda x: wavg(x, return_var, weight_var)).reset_index().rename(columns={0: 'portret'})

        return port
    tmp = df.copy()
    tmp = tmp.dropna(subset=[group_by_var, return_var, date_var, id_var], how='any')
    tmp[f'{group_by_var}_group'] = tmp.groupby(date_var)[group_by_var].transform(lambda x: pd.qcut(x, group_num, labels=False, duplicates='drop') + 1)
    if weight_var is None:
        tmp['weight'] = 1
        port = cal_group_ret(tmp, 'weight')
    port_pivot = port.pivot(index=date_var, columns=f'{group_by_var}_group', values='portret')
    port_pivot['hl'] = port_pivot[group_num] - port_pivot[1]
    port = port_pivot.melt(ignore_index=False).reset_index().rename(columns={'variable': 'group', 'value': 'portret'})
    return port