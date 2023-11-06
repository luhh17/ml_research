import pandas as pd
import numpy as np

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