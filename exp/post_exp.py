import os
import pandas as pd
import numpy as np
from post_analysis.portfolio import sorting_portfolio
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from utils.post_analysis import convert_array, backtest, balance_weight, calculate_turnover, calculate_sr, calculate_port_ret, set_number_format
pd.set_option('display.max_columns', None)

class Post_Exp(object):
    def __init__(self, args):
        self.args = args
        if args is None:
            return
        if self.args.est_method == 'rolling':
            df = pd.read_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df = pd.read_pickle(self.args.model_file_path + '/pred_result')
        df = df[df['mask'] == 1]
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        self.df = df

    def plot_prob(self, df):
        stkcd_list = df['stkcd'].unique()
        for idx, stkcd in enumerate(stkcd_list):
            _df = df[df['stkcd'] == stkcd].copy()
            values = _df.to_numpy()
            pred_mat = values[:, 2:102]
            top5 = np.percentile(pred_mat, 95, axis=1)
            top95 = np.percentile(pred_mat, 5, axis=1)
            pred = np.mean(pred_mat, axis=1)
            true = values[:, 102]
            plt.plot(true, label='true')
            plt.plot(top5, label='top5')
            plt.plot(top95, label='top95')
            plt.plot(pred, label='pred')
            plt.title(stkcd)
            plt.legend()
            plt.show()


    def keyu_corr(self, _df=None):
        data_path = '/mnt/SSD4TB_1/data_output'
        return_file = 'Ashare_Stock_raw.pkl'
        market_weight_file = 'CSI_500_weight.pkl'
        # result_path = '/mnt/SSD4TB_1/Barra/ipca'
        # Variable names
        date_var = 'date'
        stock_var = 'stkcd'
        ret_var = 'ret_open'

        ###############################################################################
        # Read data
        market_weight_df = pd.read_pickle(os.path.join(data_path, market_weight_file))
        market_weight_df.set_index([date_var, stock_var], inplace=True)

        ###############################################################################
        # Calculate for each result
        if _df is None:
            _df = self.df.copy()
        _df = _df[_df['mask'] == 1]
        _df['date'] = pd.to_datetime(_df['date'], format='%Y%m%d')
        _df = _df[_df['mask'] == 1]
        _df['stkcd'] = _df['stkcd'].astype(str).str.zfill(6)
        _df = _df.set_index(['date', 'stkcd'])
        _df = _df[_df.index.isin(market_weight_df.index)]
        _df = _df.reset_index()

        sample_count = _df.groupby('stkcd')['mask'].agg('sum').reset_index()
        sample_stocks = sample_count[sample_count['mask'] > _df['date'].nunique() * 0.5]
        _df = pd.merge(_df, sample_stocks, on='stkcd', how='inner')
        _df['resid'] = _df['ret'] - _df['pred']
        resid_matrix = _df.pivot(index='date', columns='stkcd', values='resid')
        corr = resid_matrix.corr()
        iret_corr_abs_mean = np.sum(np.triu(np.abs(corr), k=1)) / (corr.shape[0] * (corr.shape[0] - 1) / 2)
        return iret_corr_abs_mean

    # 计算ivol部分的相关性，有两种计算方式： 1. 在每个截面直接ivol相乘，基于期望为0的假设 2. 计算ivol序列的历史相关性
    def ivol_corr(self, df=None):
        if df is None:
            df = self.df.copy()
        df = df[df['mask'] == 1]
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        df['resid'] = df['ret'] - df['pred']
        resid_matrix = df.pivot(index='date', columns='stkcd', values='resid').values
        ivol_matrix = np.expand_dims(resid_matrix, axis=-1) @ np.expand_dims(resid_matrix, axis=1)
        ivol_matrix[:, np.arange(ivol_matrix.shape[1]), np.arange(ivol_matrix.shape[1])] = np.nan
        ivol_matrix = np.abs(ivol_matrix)
        cs_ivol = np.nanmean(ivol_matrix, axis=(1, 2))
        mean_ivol = np.mean(cs_ivol)

        # method 2: cal corr of resid series
        resid_matrix = df.pivot(index='date', columns='stkcd', values='resid')
        corr = resid_matrix.corr().values
        corr[np.arange(corr.shape[0]), np.arange(corr.shape[0])] = np.nan
        abs_corr = np.abs(corr)
        mean_corr = np.nanmean(abs_corr)

        ret_matrix = df.pivot(index='date', columns='stkcd', values='ret')
        corr = ret_matrix.corr().values
        corr[np.arange(corr.shape[0]), np.arange(corr.shape[0])] = np.nan
        abs_corr = np.abs(corr)
        raw_ret_corr = np.nanmean(abs_corr)

        return mean_ivol, mean_corr, raw_ret_corr

    def beta_autocorr(self, df=None):
        if df is None:
            df = self.df.copy()
        corr_list = []
        for i in range(self.args.factor_num):
            beta_matrix = df.pivot(index='date', columns='stkcd', values=f'pred_beta{i}')
            s = beta_matrix.apply(lambda x: x.autocorr(), axis=0)
            beta_corr = np.nanmean(np.abs(s))
            corr_list.append(beta_corr)
        return np.mean(corr_list)

    def factor_analysis(self, df=None):
        if df is None:
            df = self.df.copy()
        keyu_corr = self.keyu_corr(df)
        port = None
        agg_sr = None
        mean_ivol, mean_corr, raw_ret_corr = self.ivol_corr(df)
        beta_autocorr = self.beta_autocorr(df)
        if self.args.nolead:
            return keyu_corr, mean_ivol, mean_corr, raw_ret_corr, beta_autocorr
        # if not self.args.nolead:
        #     port = []
        #     agg_sr = []
        #     for i in range(self.args.factor_num):
        #         port_ret, sr = self.sort_return(f'pred_beta{i}', df)
        #         port.append(port_ret)
        #         agg_sr.append(sr)
        #     port = pd.concat(port, axis=1)
        #     port.columns = [f'beta{i}_return' for i in range(self.args.factor_num)]
        #     agg_sr = pd.concat(agg_sr, axis=1)
        #     agg_sr.columns = [f'beta{i}_sr' for i in range(self.args.factor_num)]
        return keyu_corr, mean_ivol, mean_corr, raw_ret_corr, port, agg_sr, beta_autocorr

    def sort_return(self, sort_var, df=None):
        if df is None:
            df = self.df.copy()
        df = df[df['mask'] == 1]
        df = df.sort_values(by=['date', 'stkcd'])
        unique = df.groupby('date')[sort_var].nunique()
        unique = unique[unique > 100].reset_index().rename(columns={sort_var: 'count'})
        # print(unique)
        df = pd.merge(df, unique, on='date', how='inner')
        sort_port = sorting_portfolio(df, sort_var)
        grouped = sort_port.groupby(['date', f'{sort_var}_group'])['portret'].mean().reset_index()
        grouped = grouped.groupby(f'{sort_var}_group')['portret'].mean()
        sr = grouped / sort_port.groupby(f'{sort_var}_group')['portret'].std()
        return grouped, sr


    def sorting_portfolio(self):
        mse = np.mean(np.square(self.df['pred'] - self.df['ret']))
        df = self.df.sort_values(by=['date', 'stkcd'])
        df['lag_pred'] = df.groupby('stkcd')['pred'].shift(1)
        print('mse', mse)
        ic = df['pred'].corr(df['ret'])
        print('corr between pred and ret', ic)
        accuracy = ((df['pred'] * df['ret']) > 0).sum() / df.shape[0]
        print('prediction accuracy', accuracy)
        sort_port = sorting_portfolio(df, 'pred')
        grouped = sort_port.groupby(['date', 'pred_group'])['portret'].mean().reset_index()
        grouped = grouped.groupby('pred_group')['portret'].mean()
        print('monthly return', grouped * 20)
        sr = grouped / sort_port.groupby('pred_group')['portret'].std() * np.sqrt(252)
        print('annual SR', sr)
        return mse, ic

    def analyze_attention(self):
        atten_list = pkl.load(open(f'{self.args.model_file_path}/atten', 'rb'))
        mean_atten = np.mean(atten_list, axis=0)
        print(mean_atten.shape)
        mean_atten = np.mean(mean_atten, axis=0)
        print(mean_atten.shape)
        print(mean_atten)
        # atten_list = np.concatenate([np.concatenate(a, axis=0) for a in atten_list], axis=0).astype('float16').round(2)
        print('atten list shape', atten_list.shape)
        atten_list = atten_list.reshape(-1)
        print('attn mean', atten_list.mean())
        print('attn std', atten_list.std())
        print('attn max', atten_list.max())
        print('attn min', atten_list.min())

        plt.figure()
        plt.hist(atten_list, bins=200)
        plt.savefig(f'{self.args.model_file_path}/atten_hist.png')
        plt.close()

        # plt.figure()
        # sns.kdeplot(atten_list)
        # plt.savefig(f'{self.args.model_file_path}/atten_kde.png')
        # plt.close()

    def backtest(self, date_var='date', ret_var='ret_open', id_var='stkcd'):
        result_dfs = []
        mean_by_year = []
        sr_by_year = []
        to_by_year = []

        pred_df = []
        file_list = os.listdir(self.args.model_file_path)
        pred_file_list = [f for f in file_list if f.endswith('_pred_result')]
        for pred in pred_file_list:
            df = pd.read_pickle(f'{self.args.model_file_path}/{pred}')
            pred_df.append(df)
        pred_df = pd.concat(pred_df, axis=0)
        pred_df = pred_df.drop(columns=['ret'])
        pred_df['pred'] /= 100
        pred_df[date_var] = pd.to_datetime(pred_df[date_var], format='%Y%m%d')
        pred_df = pred_df[pred_df['mask'] == True]
        trading_adv_df = pd.read_pickle('/mnt/SSD4TB_1/data_output/trading_adv_info.pkl')
        return_df = pd.read_pickle('/mnt/SSD4TB_1/data_output/Ashare_Stock_return.pkl')
        market_df = pd.read_pickle('/mnt/SSD4TB_1/data_output/CSI_500.pkl')
        return_df = pd.merge(return_df, market_df, on=date_var, how='left')
        return_df['exret'] = return_df[ret_var] - return_df['mktret']
        ret_adv_df = pd.merge(return_df, trading_adv_df, on=[date_var, id_var], how='inner')
        market_df.set_index(date_var, inplace=True)
        pred_df.rename(columns={'pred': 'Raw'}, inplace=True)
        if pred_df[id_var].dtype != 'str':
            pred_df[id_var] = pred_df[id_var].astype(str).str.zfill(6)
        # Combine and pre-process data
        predict_df = pd.merge(pred_df, ret_adv_df, on=[date_var, id_var], how='inner')
        date_series = predict_df[date_var].unique()
        predict_df.set_index([date_var, id_var], inplace=True)
        mkt_df_subset = market_df.loc[date_series]

        array, mask, _, columns = convert_array(predict_df)
        signal = array[..., columns.get_loc('Raw')].astype(float)
        ret = array[..., columns.get_loc(ret_var)].astype(float)
        stock_adv = array[..., columns.get_loc('adv')].astype(float)
        mask_buy = array[..., columns.get_loc('mask_buy')].astype(bool) * mask
        mask_sell = array[..., columns.get_loc('mask_sell')].astype(bool) * mask
        market = mkt_df_subset['mktret'].to_numpy()
        stock_adv_limit = self.args.adv_limit * stock_adv
        stock_adv_limit_hold = self.args.adv_limit_hold * stock_adv

        ret_rb15_bc, turnover_rb15_bc, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                    mask_sell, self.args.capital_tot,
                                                    shorting=True, market=market, rebalance=1.5, cost_rate=0,
                                                    std_cutoff=self.args.std_cutoff)
        ret_rb15_ac, turnover_rb15_ac, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                    mask_sell, self.args.capital_tot,
                                                    shorting=True, market=market, rebalance=1.5,
                                                    cost_rate=self.args.trading_cost_rate, std_cutoff=self.args.std_cutoff)
        ret_rb1_bc, turnover_rb1_bc, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                  mask_sell, self.args.capital_tot,
                                                  shorting=True, market=market, rebalance=1, cost_rate=0,
                                                  std_cutoff=self.args.std_cutoff)
        ret_rb1_ac, turnover_rb1_ac, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                  mask_sell, self.args.capital_tot,
                                                  shorting=True, market=market, rebalance=1,
                                                  cost_rate=self.args.trading_cost_rate, std_cutoff=self.args.std_cutoff)
        ret_long_bc, turnover_long_bc, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                    mask_sell, self.args.capital_tot,
                                                    cost_rate=0, std_cutoff=self.args.std_cutoff)
        ret_long_ac, turnover_long_ac, _ = backtest(signal, ret, stock_adv_limit, stock_adv_limit_hold, mask_buy,
                                                    mask_sell, self.args.capital_tot,
                                                    cost_rate=self.args.trading_cost_rate, std_cutoff=self.args.std_cutoff)

        # Balance weights
        predict_df['Balanced'] = balance_weight(predict_df, 'Raw')
        predict_df['Long'] = predict_df['Balanced'] * (predict_df['Balanced'] > 0)
        predict_df['Short'] = predict_df['Balanced'] * (predict_df['Balanced'] < 0)

        # Calculate size and adv average ranks
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

        # Calculate statistics
        turnover_df = calculate_turnover(predict_df[['Raw', 'Balanced', 'Long', 'Short']])
        turnover_df['ADV_rb1.5_bc'] = turnover_rb15_bc
        turnover_df['ADV_rb1.5_ac'] = turnover_rb15_ac
        turnover_df['ADV_rb1_bc'] = turnover_rb1_bc
        turnover_df['ADV_rb1_ac'] = turnover_rb1_ac
        turnover_df['ADV_long_bc'] = turnover_long_bc
        turnover_df['ADV_long_ac'] = turnover_long_ac
        port_ret_df = calculate_port_ret(predict_df[['exret', 'Raw', 'Balanced', 'Long', 'Short']], 'exret')
        port_ret_df['ADV_rb1.5_bc'] = ret_rb15_bc
        port_ret_df['ADV_rb1.5_ac'] = ret_rb15_ac
        port_ret_df['ADV_rb1_bc'] = ret_rb1_bc
        port_ret_df['ADV_rb1_ac'] = ret_rb1_ac
        port_ret_df['ADV_long_bc'] = ret_long_bc - market
        port_ret_df['ADV_long_ac'] = ret_long_ac - market
        result_df = calculate_sr(port_ret_df)
        result_df.loc['TO'] = turnover_df.mean()
        result_df.loc['Size_rank', ['Raw', 'Balanced', 'Long', 'Short']] = size_avg_rank
        result_df.loc['ADV_rank', ['Raw', 'Balanced', 'Long', 'Short']] = adv_avg_rank
        result_by_year_df = port_ret_df.groupby(port_ret_df.index.year).apply(calculate_sr)
        to_by_year_df = turnover_df.groupby(turnover_df.index.year).mean()

        # Add results
        # result_dfs.append(result_df)
        # mean_by_year.append(result_by_year_df.loc[:, 'mean', :])
        # sr_by_year.append(result_by_year_df.loc[:, 'SR', :])
        # to_by_year.append(to_by_year_df)
        mean_by_year = result_by_year_df.loc[:, 'mean', :]
        sr_by_year = result_by_year_df.loc[:, 'SR', :]
        to_by_year = to_by_year_df

        # Combine results
        # result = pd.concat(result_dfs, keys=predict_files.keys(), names=['Model', 'Stat'])
        # mean_by_year = pd.concat(mean_by_year, keys=predict_files.keys(), names=['Model', 'year'])
        # sr_by_year = pd.concat(sr_by_year, keys=predict_files.keys(), names=['Model', 'year'])
        # to_by_year = pd.concat(to_by_year, keys=predict_files.keys(), names=['Model', 'year'])

        # Save results
        result_save_file = os.path.join(self.args.model_file_path, 'agg_result.xlsx')
        with pd.ExcelWriter(result_save_file) as writer:
            result_df.to_excel(writer, sheet_name='result')
            mean_by_year.to_excel(writer, sheet_name='Mean by year')
            sr_by_year.to_excel(writer, sheet_name='SR by year')
            to_by_year.to_excel(writer, sheet_name='TO by year')

            # Set number format
            n_rows = mean_by_year.shape[0]
            n_columns = result_df.shape[1]
            for i in range(1):
                set_number_format(writer.sheets['result'], ((6 * i + 2, 3), (6 * i + 4, 3 + n_columns)), '0.00%')
                set_number_format(writer.sheets['result'], ((6 * i + 4, 3), (6 * i + 8, 3 + n_columns)), '0.00')
            set_number_format(writer.sheets['Mean by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00%')
            set_number_format(writer.sheets['SR by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00')
            set_number_format(writer.sheets['TO by year'], ((2, 3), (2 + n_rows, 3 + n_columns)), '0.00')

            # Set auto column width
            writer.sheets['result'].column_dimensions['A'].auto_size = True
            writer.sheets['Mean by year'].column_dimensions['A'].auto_size = True
            writer.sheets['SR by year'].column_dimensions['A'].auto_size = True
            writer.sheets['TO by year'].column_dimensions['A'].auto_size = True

