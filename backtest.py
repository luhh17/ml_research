import os
import pickle as pkl
import pandas as pd
import numpy as np
import yaml
import sys
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from utils.post_analysis import convert_array, backtest, balance_weight, calculate_turnover, calculate_sr, calculate_port_ret, set_number_format
import pdb
from sklearn.linear_model import LinearRegression
from post_analysis.get_one_model_pred import load_pred_result



class Backtest():
    def __init__(self):
        args = yaml.safe_load(open("./backtest_config.yml", 'r'))
        self.args = args
        self.config_var = ['model', 'window_length', 'pred_length', 'dec_len_from_input', 'embed_dim', 'hidden_dim',
                                                    'num_enc_layers', 'num_dec_layers', 'num_heads']
        self.adv_limit = args['adv_limit']
        self.adv_limit_hold = args['adv_limit_hold']
        self.std_cutoff = args['std_cutoff']
        self.capital_tot = float(args['capital_tot'])
        self.trading_cost_rate = args['trading_cost']
        self.period_start = args['period_start']
        self.period_end = args['period_end']
        self.date_var = 'date'
        self.id_var = 'stkcd'
        self.ret_var = 'ret_open'
        trading_adv_df = self.keep_only_topn(self.args['test_topn'])
        return_df = pd.read_pickle(f"{self.args['backtest_data_path']}/ret_decomposed.pkl")
        market_df = pd.read_pickle(f"{self.args['backtest_data_path']}/CSI_500.pkl")
        
        index_df = pd.read_pickle(f"{self.args['backtest_data_path']}/IDX_Smprat.pkl")[['Indexcd','Enddt','Stkcd','Weight']]
        index_df.rename(columns={'Enddt':'date','Stkcd':'stkcd'},inplace=True)

        return_df = pd.merge(return_df, index_df, on=[self.date_var, self.id_var], how='left')
        return_df = pd.merge(return_df, market_df, on=self.date_var, how='left')
        return_df['exret'] = return_df['ret_close'] - return_df['mktret']
        self.ret_adv_df = pd.merge(return_df, trading_adv_df, on=[self.date_var, self.id_var], how='inner')
        market_df.set_index(self.date_var, inplace=True)
        self.market_df = market_df

    
    def cal_size_rank(self, predict_df: pd.DataFrame) -> float:  
        """ 
        Calculate size and adv average ranks
        Parameters:
            predict_df: pd.DataFrame
        Returns:
            size_avg_rank: float
            adv_avg_rank: float
        """
        date_var = 'date'
        rank_df = pd.DataFrame(index=predict_df.index)
        rank_df[['Raw', 'Balanced', 'Long', 'Short']] = abs(predict_df[['Raw', 'Balanced', 'Long', 'Short']])
        rank_df['Raw'] /= rank_df['Raw'].groupby(date_var).sum()
        rank_df['Balanced'] /= 2
        rank_df['size'] = predict_df['size'].groupby(date_var).rank(ascending=False)
        rank_df['adv'] = predict_df['adv'].groupby(date_var).rank(ascending=False)
        size_avg_rank = (
            rank_df[['Raw', 'Balanced', 'Long', 'Short']] * rank_df['size'].to_numpy().reshape(-1, 1)).groupby(
            date_var).sum().mean()
        adv_avg_rank = (
            rank_df[['Raw', 'Balanced', 'Long', 'Short']] * rank_df['adv'].to_numpy().reshape(-1, 1)).groupby(
            date_var).sum().mean()
        return size_avg_rank, adv_avg_rank



    def keep_only_topn(self, topn):
        trading_adv_df = pd.read_pickle(f"{self.args['backtest_data_path']}/trading_adv_info.pkl")
        trading_adv_df['year'] = trading_adv_df['date'].dt.year
        trading_adv_df_top1000 = pd.DataFrame()
        for year in trading_adv_df['year'].unique():
            # Filter data for the current year
            year_data = trading_adv_df[trading_adv_df['year'] == year]
            first_date_of_year = year_data['date'].min()
            year_data_first_date = year_data[year_data['date'] == first_date_of_year].copy()
            year_data_first_date.loc[:,'mktcap_r'] = year_data_first_date['size'].rank(ascending=False, method='min')
            top1000_universe = year_data_first_date[year_data_first_date['mktcap_r'] <= topn]['stkcd']
            trading_adv_df_top1000 = pd.concat([trading_adv_df_top1000, year_data[year_data['stkcd'].isin(top1000_universe)]])
            del year_data, year_data_first_date
        trading_adv_df = trading_adv_df_top1000.copy()
        return trading_adv_df


    def backtest_one_model(self, pred_df):
        pred_df = pred_df[(pred_df['date'] >= self.period_start) & (pred_df['date'] <= self.period_end)]
        pred_df.rename(columns={'pred': 'Raw'}, inplace=True)
        if pred_df[self.id_var].dtype != 'str':
            pred_df[self.id_var] = pred_df[self.id_var].astype(str).str.zfill(6)
        ret_adv_df = self.ret_adv_df.copy()
        market_df = self.market_df.copy()

        # Combine and pre-process data
        predict_df = pd.merge(pred_df, ret_adv_df, on=[self.date_var, self.id_var], how='inner')
        date_series = predict_df[self.date_var].unique()
        predict_df.set_index([self.date_var, self.id_var], inplace=True)
        mkt_df_subset = market_df.loc[date_series]

        array, mask, _, columns = convert_array(predict_df)
        signal = array[..., columns.get_loc('Raw')].astype(float)
        ret = array[..., columns.get_loc(self.ret_var)].astype(float)
        stock_adv = array[..., columns.get_loc('adv')].astype(float)
        mask_buy = array[..., columns.get_loc('mask_buy')].astype(bool) * mask
        mask_sell = array[..., columns.get_loc('mask_sell')].astype(bool) * mask
        market = mkt_df_subset['mktret'].to_numpy()
        stock_adv_limit = self.adv_limit * stock_adv
        stock_adv_limit_hold = self.adv_limit_hold * stock_adv
        ret_rb15_bc, turnover_rb15_bc, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                    mask_sell, self.capital_tot,
                                                    shorting=True, market=market, rebalance=1.5, cost_rate=0,
                                                    std_cutoff=self.std_cutoff)

        ret_rb1_bc, turnover_rb1_bc, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                mask_sell, self.capital_tot,
                                                shorting=True, market=market, rebalance=1, cost_rate=0,
                                                std_cutoff=self.std_cutoff)
    
        ret_long_bc, turnover_long_bc, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                    mask_sell, self.capital_tot,
                                                    cost_rate=0, std_cutoff=self.std_cutoff)

        # Balance weights
        predict_df['Balanced'] = balance_weight(predict_df, 'Raw')
        predict_df['Long'] = predict_df['Balanced'] * (predict_df['Balanced'] > 0)
        predict_df['Short'] = predict_df['Balanced'] * (predict_df['Balanced'] < 0)
        weight_300=predict_df[predict_df['Indexcd']=='000300'].groupby(level=0)['Long'].sum()
        weight_500=predict_df[predict_df['Indexcd']=='000905'].groupby(level=0)['Long'].sum()
        weight_others=1-weight_300 - weight_500
        
        # Calculate size and adv average ranks
        size_avg_rank, adv_avg_rank = self.cal_size_rank(predict_df)

        # Calculate statistics
        turnover_df = calculate_turnover(predict_df[['Raw', 'Balanced', 'Long', 'Short']])
        turnover_df['ADV_rb1.5_bc'] = turnover_rb15_bc
        turnover_df['ADV_rb1_bc'] = turnover_rb1_bc
        turnover_df['ADV_long_bc'] = turnover_long_bc

        port_absret_df,port_ret_df = calculate_port_ret(predict_df[['ret_close_to_open', 'ret_open_to_close', 'mktret','Raw', 'Balanced', 'Long', 'Short']])
        port_ret_df['ADV_rb1.5_bc'] = ret_rb15_bc
        port_ret_df['ADV_rb1_bc'] = ret_rb1_bc
        port_ret_df['ADV_long_bc'] = ret_long_bc - market
        

        index_ret = pd.read_csv(f"{self.args['backtest_data_path']}/indexret.csv",index_col='date')
        index_ret.index = pd.to_datetime(index_ret.index)
        index_ret = index_ret.loc[port_ret_df.index,:]
        mkt_ret = market_df.loc[port_ret_df.index,:]

        long_port_df = pd.concat([port_absret_df['Long'], mkt_ret, index_ret, weight_300, weight_500, weight_others], axis=1)
        long_port_df.columns = ['bc_long_daily_ret','mkt_ret','300_daily_ret','500_daily_ret','300_weight','500_weight','others_weight']

        result_df = calculate_sr(port_ret_df)
        result_df.loc['TO'] = turnover_df.mean()
        result_df.loc['Size_rank', ['Raw', 'Balanced', 'Long', 'Short']] = size_avg_rank
        result_df.loc['ADV_rank', ['Raw', 'Balanced', 'Long', 'Short']] = adv_avg_rank
        result_by_year_df = port_ret_df.groupby(port_ret_df.index.year).apply(calculate_sr)
        to_by_year_df = turnover_df.groupby(turnover_df.index.year).mean()

        return result_df, turnover_df, to_by_year_df, port_ret_df, result_by_year_df, long_port_df


    def output_backtest_to_excel(self, result_save_file: str, result, mean_by_year, sr_by_year, dd_by_year, to_by_year, model_list):
        with pd.ExcelWriter(result_save_file, engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name='result')
            mean_by_year.to_excel(writer, sheet_name='Mean by year')
            sr_by_year.to_excel(writer, sheet_name='SR by year')
            dd_by_year.to_excel(writer, sheet_name='Drawdown by year')
            to_by_year.to_excel(writer, sheet_name='TO by year')
            # Set number format
            n_models = len(model_list)
            n_rows = mean_by_year.shape[0]
            n_columns = result.shape[1]
            for i in range(n_models):
                set_number_format(writer.sheets['result'], ((9 * i + 2, 3), (9 * i + 4, 3 + n_columns)), '0.00%')
                set_number_format(writer.sheets['result'], ((9 * i + 4, 3), (9 * i + 5, 3 + n_columns)), '0.00')
                set_number_format(writer.sheets['result'], ((9 * i + 5, 3), (9 * i + 6, 3 + n_columns)), '0.00%')
                set_number_format(writer.sheets['result'], ((9 * i + 6, 3), (9 * i + 11, 3 + n_columns)), '0.00')
            set_number_format(writer.sheets['Mean by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00%')
            set_number_format(writer.sheets['SR by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00')
            set_number_format(writer.sheets['Drawdown by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00%')
            set_number_format(writer.sheets['TO by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00')

            # Set auto column width
            writer.sheets['result'].column_dimensions['A'].auto_size = True
            writer.sheets['Mean by year'].column_dimensions['A'].auto_size = True
            writer.sheets['SR by year'].column_dimensions['A'].auto_size = True
            writer.sheets['Drawdown by year'].column_dimensions['A'].auto_size = True
            writer.sheets['TO by year'].column_dimensions['A'].auto_size = True


    def output_daily_long_port_to_excel(self, long_port_df_list: list, output_model_list: list, port_result_save_file: str):
        with pd.ExcelWriter(port_result_save_file) as writer1:
            for i, long_port_df in enumerate(long_port_df_list):
                long_port_df.to_excel(writer1, sheet_name=output_model_list[i])
                set_number_format(writer1.sheets[output_model_list[i]], ((1, 1), (long_port_df.shape[0]+1, 2)), 'yyyy-mm-dd')
                set_number_format(writer1.sheets[output_model_list[i]], ((1, 2), (long_port_df.shape[0]+1, 9)), '0.00%')



# 计算模型后的结果
    def backtest_metric(self, num_model=1, date_var='date', ret_var='ret_open', id_var='stkcd'):
        """
        Calculate the result of ensemble models, automatically output to excel
        Parameters:
            path_name: str
            num_model: int
            strategy: str
            date_var: str
            ret_var: str
            id_var: str
        Returns:
            None
        """
        args = self.args
        model_root_path = args['model_path']
        dict_name = f"{args['report_output_path']}/{args['test_topn']}_res_dict.pkl"
        output_name = f"{args['report_output_path']}/{args['test_topn']}_result.xlsx"
        port_ret_name= f"{args['report_output_path']}/{args['test_topn']}_result_longport_ret.xlsx"
        model_list = np.sort(os.listdir(model_root_path))

        model_name_path_dict = {model_name: os.path.join(model_root_path, model_name) for model_name in model_list}

        long_port_dfs = []
        result_dfs = []
        mean_by_year = []
        sr_by_year = []
        dd_by_year = []
        to_by_year = []
        output_model_list = []

        if not os.path.exists(args['report_output_path']):
            os.makedirs(args['report_output_path'])

        if os.path.exists(f"{dict_name}"):
            res_dict = pkl.load(open(f"{dict_name}", 'rb'))
        else:
            res_dict = {}

        for path in model_name_path_dict.values():
            if path.split('/')[-1] in list(res_dict.keys()):
                continue
            print(path)
            model_idx = np.sort(os.listdir(path))
            model_idx = [m for m in model_idx if os.path.isdir(os.path.join(path, m))]
            pred_list = []
            if len(model_idx) == 0:
                continue
            for idx in model_idx:
                pred_df = load_pred_result(path + f'/{idx}')
                if pred_df is None:
                    continue
                pred = pred_df['pred'].values
                pred_list.append(pred)
            if len(pred_list) == 0:
                continue

            output_model_list.append(path.split('/')[-1])
            pred_list = np.array(pred_list)

            pred_list = np.mean(pred_list, axis=0)
            pred_df['pred'] = pred_list
            result_df, turnover_df, to_by_year_df, port_ret_df, result_by_year_df, long_port_df = self.backtest_one_model(pred_df)
            # Add results
            long_port_dfs.append(long_port_df)
            result_dfs.append(result_df)
            mean_by_year.append(result_by_year_df.loc[:, 'mean', :])
            sr_by_year.append(result_by_year_df.loc[:, 'SR', :])
            dd_by_year.append(result_by_year_df.loc[:, 'max_dd', :])
            to_by_year.append(to_by_year_df)

        per_model_df = [[result_dfs[i], mean_by_year[i], sr_by_year[i], dd_by_year[i], to_by_year[i]] for i in range(len(output_model_list))]
        if len(output_model_list) != 0:
            new_res_dict = dict(zip(output_model_list, per_model_df))
            res_dict.update(new_res_dict)
        pkl.dump(res_dict, open(f"{dict_name}", 'wb'))
        mean_by_year = [v[1] for v in list(res_dict.values())]
        sr_by_year = [v[2] for v in list(res_dict.values())]
        dd_by_year=[v[3] for v in list(res_dict.values())]
        to_by_year = [v[4] for v in list(res_dict.values())]
        output_model_list = list(res_dict.keys())
        # Combine results

        result = pd.concat(result_dfs, keys=output_model_list, names=['Model', 'Stat'])
        mean_by_year = pd.concat(mean_by_year, keys=output_model_list, names=['Model', 'year'])
        sr_by_year = pd.concat(sr_by_year, keys=output_model_list, names=['Model', 'year'])
        dd_by_year = pd.concat(dd_by_year, keys=output_model_list, names=['Model', 'year'])
        to_by_year = pd.concat(to_by_year, keys=output_model_list, names=['Model', 'year'])

        result_save_file = os.path.join(f"{output_name}")
        self.output_backtest_to_excel(result_save_file, result, mean_by_year, sr_by_year, dd_by_year, to_by_year, output_model_list)




if __name__ == '__main__':
    args = yaml.safe_load(open("./backtest_config.yml", 'r'))
    backtest_block = Backtest()
    backtest_block.backtest_metric(args)
