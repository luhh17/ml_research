import pandas as pd
import numpy as np
# from utils.logger import build_model_file_path_from_config
import yaml

def load_true_ret():
    date_list = ["2017-12-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
    train_start_date = date_list[0]
    test_end_date = date_list[-1]
    print(train_start_date, test_end_date)
    fea_index = pd.read_pickle('./../data_annual_1113/index.pkl')
    sel_index = fea_index[(fea_index['date']>= train_start_date) & (fea_index['date']<= test_end_date)].index
    data = pd.read_hdf('./../data_annual_1113/signal_rank.h5', start=sel_index[0], stop=sel_index[-1]+1)
    data = data[['stkcd', 'date', 'ret_open', 'exret_open', 'C.1.1']]
    print(data.head())
    print(data.shape)
    # No extreme return
    data_filter = data[np.abs(data['ret_open'])<0.25].reset_index(drop=True)
    # top 2000 universe
    data_filter['year'] = data_filter['date'].dt.year
    ret_data_list = []
    for year in data_filter['year'].unique():
        year_data = data_filter[data_filter['year'] == year]
        first_date_of_year = year_data['date'].min()
        year_data_first_date = year_data[year_data['date'] == first_date_of_year].copy()
        year_data_first_date.loc[:,'mktcap_r'] = year_data_first_date['C.1.1'].rank(ascending=False, method='min')
        ret_data = pd.merge(year_data[['stkcd', 'date', 'year', 'ret_open', 'exret_open']], 
        year_data_first_date[['stkcd', 'year', 'mktcap_r']], on=['stkcd', 'year'], how='left')
        ret_data_list.append(ret_data)
    ret_data = pd.concat(ret_data_list, axis=0)
    ret_data = ret_data.sort_values(by=['stkcd', 'date'])
    ret_data['3d_open'] = ret_data.groupby('stkcd')['ret_open'].rolling(3).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(0, drop=True)
    ret_data['3d_exret'] = ret_data.groupby('stkcd')['exret_open'].rolling(3).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(0, drop=True)
    ret_data['5d_open'] = ret_data.groupby('stkcd')['ret_open'].rolling(5).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(0, drop=True)
    ret_data['5d_exret'] = ret_data.groupby('stkcd')['exret_open'].rolling(5).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(0, drop=True)
    ret_data['10d_open'] = ret_data.groupby('stkcd')['ret_open'].rolling(10).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(0, drop=True)
    ret_data['10d_exret'] = ret_data.groupby('stkcd')['exret_open'].rolling(10).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(0, drop=True)
    ret_data.to_pickle('./../data_annual_1113/true_ret.pkl')
    return ret_data
    

def ols_ensemble(realized_ret: pd.DataFrame, df_list: list, date_var='date', ret_var='ret_open', id_var='stkcd') -> pd.DataFrame:
    """
    Using OLS to ensemble the prediction results from different models
    Y is realized return
    X is the prediction results from different models
    And we only return the fitted results minus alpha
    Parameters:
        realized_ret: pd.DataFrame
        df_list: list
        date_var: str
        ret_var: str
        id_var: str
    Returns:
        fitted_df: pd.DataFrame
    """
    x_df = pd.DataFrame()
    for df in df_list:
        df.set_index([date_var, id_var], inplace=True)
        df = df.rename(columns={'pred': df.name})
        x_df = pd.concat([x_df, df], axis=1)
    x_df = x_df.reset_index()
    x_df = x_df.merge(realized_ret, on=[date_var, id_var], how='inner')
    x_df = x_df.dropna()
    reg = LinearRegression(fit_intercept=True).fit(x_df.iloc[:, 2:-1], x_df[ret_var])
    alpha = reg.intercept_
    pred = reg.predict(x_df.iloc[:, 2:-1]) - alpha
    return pred



def agg_one_model_pred():
    backtest_args = yaml.safe_load(open("./../infer_config.yml", 'r'))
    for config in backtest_args['config_list']:
        args = yaml.safe_load(open(config, 'r'))
        task_name, model_file_path = build_model_file_path(args)
        for idx in args['ensemble_num']:
            pred_df = load_pred_result(model_file_path + f'/{idx}')
            print(pred_df.head())
        if not pred_df or pred_df.empty:
            continue


if __name__ == '__main__':
    load_true_ret()
    # ret_data = pd.read_pickle('./../data_annual_1113/true_ret.pkl')
    
    # ols_ensemble()
    # agg_one_model_pred()