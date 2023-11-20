from utils.logger import build_model_file_path_from_config
import yaml
from post_analysis.get_one_model_pred import load_pred_result
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle as pkl
from backtest import Backtest

class ModelEnsemble():
    def __init__(self, suffix):
        ensemble_args = yaml.safe_load(open("./infer_config.yml", 'r'))
        self.ensemble_args = ensemble_args
        self.true_ret = pd.read_pickle('./../data_annual_1113/true_ret.pkl')
        self.suffix = suffix

    def concat_pred_result(self):
        concat_pred = pd.DataFrame()
        ret_type = []
        for config in self.ensemble_args['config_list']:
            args = yaml.safe_load(open(config, 'r'))
            ret_type.append(f"{args['ret_type']}{args['target_window']}d")
            task_name, model_file_path = build_model_file_path_from_config(args)
            pred_array = []
            for idx in range(args['ensemble_num']):
                pred_df = load_pred_result(model_file_path + f'/{idx}', self.suffix)
                pred_array.append(pred_df['pred'].values)
            pred_array = np.array(pred_array)
            pred_array = np.mean(pred_array, axis=0)
            pred_df = pd.DataFrame({'date': pred_df['date'], 'stkcd': pred_df['stkcd'], args['model']: pred_array})
            if (pred_df is None) or pred_df.empty:
                continue
            if concat_pred.empty:
                concat_pred = pred_df
            else:
                concat_pred = concat_pred.merge(pred_df, on=['date', 'stkcd'], how='inner')
        ret_type = set(ret_type)
        if len(ret_type) != 1:
            raise ValueError('The return type is not consistent')
        return concat_pred, list(ret_type)[0]


    def est_ols_param(self, realized_ret: pd.DataFrame, concat_df: pd.DataFrame, date_var='date', ret_var='exret_open', id_var='stkcd') -> pd.DataFrame:
        """
        Using OLS to ensemble the prediction results from different models
        Y is realized return
        X is the prediction results from different models
        And we only return the fitted results minus alpha
        Parameters:
            realized_ret: pd.DataFrame
            concat_df: pd.DataFrame
            date_var: str
            ret_var: str
            id_var: str
        Returns:
            fitted_df: pd.DataFrame
        """
        x_var = [col for col in concat_df.columns if col not in [date_var, id_var, ret_var]]
        x_df = pd.merge(realized_ret, concat_df, on=[date_var, id_var], how='inner')
        x_df = x_df.dropna()
        year_list = []
        x_df['ym'] = x_df['date'].dt.year * 100 + x_df['date'].dt.month
        ym_list = np.sort(x_df['ym'].unique())
        for idx, year in enumerate(ym_list):
            year_df = x_df[x_df['ym'] == year].copy()
            reg = LinearRegression(fit_intercept=True)
            reg.fit(year_df[x_var], year_df[ret_var])
            if idx == len(ym_list) - 1:
                break
            next_year_df = x_df[x_df['ym'] == ym_list[idx + 1]].copy()
            next_year_df['pred'] = reg.predict(next_year_df[x_var]) - reg.intercept_
            year_list.append(next_year_df)
        x_df = pd.concat(year_list, axis=0)
        backtest = Backtest()
        print(x_df[x_var].mean(axis=1).shape)
        print('avg mse:', np.mean((x_df[x_var].mean(axis=1) - x_df[ret_var]) ** 2))
        print('ols ensemble mse:', np.mean((x_df[ret_var] - x_df['pred']) ** 2))
        res = backtest.backtest_one_model(x_df[['date', 'stkcd', 'pred']])
        print(res[0])
        exit()
        reg = LinearRegression(fit_intercept=True)
        reg.fit(x_df[x_var], x_df[ret_var])
        pkl.dump(reg, open('./../ensemble_param/ols_model.pkl', 'wb'))
        return


    def ols_fit(self, concat_df, date_var='date', ret_var='ret_open', id_var='stkcd') -> pd.DataFrame:
        x_var = [col for col in concat_df.columns if col not in [date_var, id_var, ret_var]]
        reg = pkl.load(open('./../ensemble_param/ols_model.pkl', 'rb'))
        alpha = reg.intercept_
        fitted = reg.predict(concat_df[x_var]) - alpha
        concat_df['pred'] = fitted
        return concat_df[['date', 'stkcd', 'pred']]


    def ensemble_via_ols(self):
        concat_pred, ret_type = self.concat_pred_result()
        self.est_ols_param(self.true_ret, concat_pred, ret_var=ret_type)
        fitted = self.ols_fit(concat_pred)
        backtest = Backtest()
        res = backtest.backtest_one_model(fitted)
        print(res[0])




if __name__ == '__main__':
    ensemble = ModelEnsemble()
    ensemble.ensemble_via_ols()

