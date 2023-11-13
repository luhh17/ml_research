import numpy as np
import pandas as pd
import torch
import time
import pickle as pkl
import pdb
from torch.utils.data import Dataset, DataLoader
from utils import data_organization
import itertools
from torch_package import functions
# index_file = 'index.pkl'
# # The data files, each file is 53 - 55 GB
# # missing filled with mean
# # all : 不加筛选
# # top2000 : 市值前2000
# # rank: rank标准化
# data_path = '/mnt/SSD4TB_1/data_output'
# file_list = ['data_exmkt_nontb_top2000_rank.h5', 'data_exmkt_nontb_top2000_norm.h5']
# index_list = ['index_exmkt_nontb_top2000_rank.pkl', 'index_exmkt_nontb_top2000_norm.pkl']


# def split_signal_return_mask(data_array, columns, data_mask):
#     input_CMat_t = np.nan_to_num(data_array[..., 2:], copy=False)
#     input_RMat_t = data_array[..., columns.get_loc('ret_open')]
#     input_IMat_t = data_mask.astype('bool')
#     input_CMat_t, input_RMat_t, input_IMat_t = torch.from_numpy(input_CMat_t), torch.from_numpy(
#         input_RMat_t), torch.from_numpy(input_IMat_t)
#     return input_CMat_t, input_RMat_t, input_IMat_t

# def split_signal_return_mask_factor(data_array, columns, data_mask):
#     input_CMat_t = np.nan_to_num(data_array[..., 2:2+1679], copy=False)
#     print(input_CMat_t.shape)
#     input_RMat_t = data_array[..., columns.get_loc('ret_open')]
#     input_IMat_t = data_mask.astype('bool')
#     factor_Matrix = np.nan_to_num(data_array[..., 2+1679:], copy=False)
#     print(factor_Matrix.shape)
#     input_CMat_t, input_RMat_t, input_IMat_t, factor_Matrix = torch.from_numpy(input_CMat_t), torch.from_numpy(
#         input_RMat_t), torch.from_numpy(input_IMat_t), torch.from_numpy(factor_Matrix)
#     return input_CMat_t, input_RMat_t, input_IMat_t, factor_Matrix

# def ts_norm(df, features):
#     df = df.sort_values(by=['date'])
#     notna = df[features].notna().to_numpy().all(axis=1)
#     first_idx = np.where(notna)
#     print(df['stkcd'])
#     print(first_idx[0])
#     print(first_idx[0][0])
#     # mean = first_hundred[features].agg(np.nanmean)
#     # std = first_hundred[features].agg(np.nanstd)
#     # df[features] = (df[features] - mean) / std
#     # df.iloc[:100, :] = np.nan
#     # df = df.iloc[100:]
#     return df


# def create_no_lead_return(stock_data):
#     stock_data = stock_data.sort_values(by=['stkcd', 'date'])
#     stock_data['ret_l1'] = stock_data.groupby('stkcd')['ret_open'].shift(-1)
#     stock_data.drop(columns=['ret_open'], inplace=True)
#     stock_data.rename(columns={'ret_l1': 'ret_open'}, inplace=True)
#     return stock_data

# def divide_train_valid_test_ff3(args):
#     data_path = '/mnt/SSD4TB_1/data_output'
#     stock_data_path = f'/mnt/HDD16TB/huahao/portfolio_construction/data/data_exmkt_nontb_top2000_rank_ff3'
#     signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')

#     features = signal_list['signal'].values.tolist()
#     features.remove('C.3.8.1')
#     to_f32 = {signal: 'float32' for signal in features}
#     stock_data = pd.read_pickle(stock_data_path)
#     stock_data = stock_data.astype(to_f32)

#     train_data = stock_data[(stock_data['date'] >= args.lag_train_start_date) & (stock_data['date'] <= args.train_end_date)]
#     del stock_data
#     train_data = create_no_lead_return(train_data)
#     train_data = train_data[(train_data['date'] >= args.train_start_date) & (train_data['date'] <= args.train_end_date)]
#     train_data.set_index(['stkcd', 'date'], inplace=True)
#     train_array, train_mask, _, train_columns, train_date, train_stkcd = data_organization.convert_array(train_data)
#     del train_data
#     train_signal, train_return, train_mask, train_factor = split_signal_return_mask_factor(train_array, train_columns, train_mask)

#     stock_data = pd.read_pickle(stock_data_path)
#     stock_data = stock_data.astype(to_f32)
#     valid_data = stock_data[
#         (stock_data['date'] >= args.lag_valid_start_date) & (stock_data['date'] <= args.valid_end_date)]
#     del stock_data
#     valid_data = create_no_lead_return(valid_data)
#     valid_data = valid_data[
#         (valid_data['date'] >= args.valid_start_date) & (valid_data['date'] <= args.valid_end_date)]
#     valid_data.set_index(['stkcd', 'date'], inplace=True)
#     valid_array, valid_mask, _, valid_columns, valid_date, valid_stkcd = data_organization.convert_array(valid_data)
#     del valid_data
#     valid_signal, valid_return, valid_mask, valid_factor = split_signal_return_mask_factor(valid_array, valid_columns, valid_mask)

#     stock_data = pd.read_pickle(stock_data_path)
#     stock_data = stock_data.astype(to_f32)
#     test_data = stock_data[
#         (stock_data['date'] >= args.lag_test_start_date) & (stock_data['date'] <= args.test_end_date)]
#     del stock_data
#     test_data = create_no_lead_return(test_data)
#     test_data = test_data[
#         (test_data['date'] >= args.test_start_date) & (test_data['date'] <= args.test_end_date)]
#     test_data.set_index(['stkcd', 'date'], inplace=True)
#     test_array, test_mask, _, test_columns, test_date, test_stkcd = data_organization.convert_array(test_data)
#     del test_data
#     test_signal, test_return, test_mask, test_factor = split_signal_return_mask_factor(test_array, test_columns, test_mask)

#     train = (train_signal, train_return, train_mask, train_date, train_stkcd, train_factor)
#     valid = (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd, valid_factor)
#     test = (test_signal, test_return, test_mask, test_date, test_stkcd, test_factor)

#     test_year = args.test_start_date.split('-')[0]
#     pkl.dump((train, valid, test), open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/roll_{test_year}_ff3_nolead', 'wb'))

#     return (train_signal, train_return, train_mask, train_date, train_stkcd), \
#               (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd), \
#                 (test_signal, test_return, test_mask, test_date, test_stkcd)


# def divide_train_valid_test_nolead(args):
#     data_path = '/mnt/SSD4TB_1/data_output'
#     stock_data_path = f'/mnt/SSD4TB_1/data_output/data_top2000_filled.h5'
#     signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')
#     to_f32 = {signal: 'float32' for signal in signal_list['signal'].values}

#     features = signal_list['signal'].values.tolist()
#     features.remove('C.3.8.1')
#     stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)

#     train_data = stock_data[
#         (stock_data['date'] >= args.lag_train_start_date) & (stock_data['date'] <= args.train_end_date)]
#     del stock_data
#     # train_data = create_no_lead_return(train_data)
#     train_data = train_data[(train_data['date'] >= args.train_start_date) & (train_data['date'] <= args.train_end_date)]
#     train_data.set_index(['stkcd', 'date'], inplace=True)
#     train_array, train_mask, _, train_columns, train_date, train_stkcd = data_organization.convert_array(train_data)
#     del train_data
#     train_signal, train_return, train_mask = split_signal_return_mask(train_array, train_columns, train_mask)

#     stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)
#     valid_data = stock_data[
#         (stock_data['date'] >= args.lag_valid_start_date) & (stock_data['date'] <= args.valid_end_date)]
#     del stock_data
#     # valid_data = create_no_lead_return(valid_data)
#     valid_data = valid_data[
#         (valid_data['date'] >= args.valid_start_date) & (valid_data['date'] <= args.valid_end_date)]
#     valid_data.set_index(['stkcd', 'date'], inplace=True)
#     valid_array, valid_mask, _, valid_columns, valid_date, valid_stkcd = data_organization.convert_array(valid_data)
#     del valid_data
#     valid_signal, valid_return, valid_mask = split_signal_return_mask(valid_array, valid_columns, valid_mask)

#     stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)
#     test_data = stock_data[
#         (stock_data['date'] >= args.lag_test_start_date) & (stock_data['date'] <= args.test_end_date)]
#     del stock_data
#     # test_data = create_no_lead_return(test_data)
#     test_data = test_data[
#         (test_data['date'] >= args.test_start_date) & (test_data['date'] <= args.test_end_date)]
#     test_data.set_index(['stkcd', 'date'], inplace=True)
#     test_array, test_mask, _, test_columns, test_date, test_stkcd = data_organization.convert_array(test_data)
#     del test_data
#     test_signal, test_return, test_mask = split_signal_return_mask(test_array, test_columns, test_mask)

#     train = (train_signal, train_return, train_mask, train_date, train_stkcd)
#     valid = (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd)
#     test = (test_signal, test_return, test_mask, test_date, test_stkcd)

#     test_year = args.test_start_date.split('-')[0]
#     pkl.dump((train, valid, test), open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/roll_{test_year}_nolead.pkl', 'wb'))

#     return (train_signal, train_return, train_mask, train_date, train_stkcd), \
#               (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd), \
#                 (test_signal, test_return, test_mask, test_date, test_stkcd)

# # 为了处理逐渐增长的回看窗口，需要建立统一的面板数据，然后再进行切分
# def divide_train_valid_test_balanced(args):
#     # data_path = '/mnt/SSD4TB_1/data_output'
#     # stock_data_path = f'/mnt/SSD4TB_1/data_output/data_exmkt_nontb_top2000_rank.h5'
#     # signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')
#     # to_f32 = {signal: 'float32' for signal in signal_list['signal'].values}
#     #
#     # features = signal_list['signal'].values.tolist()
#     # features.remove('C.3.8.1')
#     # stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)
#     # stock_data = stock_data[
#     #     (stock_data['date'] >= args.train_start_date) & (stock_data['date'] <= args.test_end_date)]
#     # stock_data.set_index(['stkcd', 'date'], inplace=True)
#     # train_array, train_mask, _, train_columns, date_matrix, stkcd_matrix = data_organization.convert_array(stock_data)
#     # signal_matrix, return_matrix, mask_matrix = split_signal_return_mask(train_array, train_columns, train_mask)
#     # date_matrix = date_matrix.squeeze()
#     # stkcd_matrix = np.expand_dims(stkcd_matrix, axis=1)
#     # stkcd_matrix = np.tile(stkcd_matrix, (1, signal_matrix.shape[1]))
#     # print('signal_matrix.shape', signal_matrix.shape)
#     # print('return_matrix.shape', return_matrix.shape)
#     # print('mask_matrix.shape', mask_matrix.shape)
#     # print('date_matrix.shape', date_matrix.shape)
#     # print('stkcd_matrix.shape', stkcd_matrix.shape)
#     test_year = args.test_start_date.split('-')[0]
#     # pkl.dump((signal_matrix, return_matrix, mask_matrix, date_matrix, stkcd_matrix), open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/roll_{test_year}_balanced_full.pkl', 'wb'),
#     #          protocol=5)

#     with open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/roll_{test_year}_balanced_full.pkl', 'rb') as file:
#         signal_matrix, return_matrix, mask_matrix, date_matrix, stkcd_matrix = pkl.load(file)
#         print(date_matrix.shape)
#         print(stkcd_matrix.shape)
#         date_list = date_matrix[0]
#         # 生成训练集
#         print(date_list)
#         print(date_list.dtype)
#         train_idx = (date_list >= int(args.train_start_date.replace('-', ''))) & (date_list <= int(args.train_end_date.replace('-', '')))
#         print(train_idx)
#         print(np.datetime64(args.train_start_date))
#         train_signal = signal_matrix[:, train_idx, :]
#         train_ret = return_matrix[:, train_idx]
#         train_mask = mask_matrix[:, train_idx]
#         train_date = date_matrix[:, train_idx]
#         train_stkcd = stkcd_matrix[:, train_idx]
#         print(train_signal.shape)
#         print(train_ret.shape)
#         print(train_mask.shape)
#         print(train_date.shape)
#         print(train_stkcd.shape)
#         # validation set
#         # 生成训练集
#         valid_idx = (date_list >= int(args.valid_start_date.replace('-', ''))) & (date_list <= int(args.valid_end_date.replace('-', '')))
#         valid_signal = signal_matrix[:, valid_idx, :]
#         valid_ret = return_matrix[:, valid_idx]
#         valid_mask = mask_matrix[:, valid_idx]
#         valid_date = date_matrix[:, valid_idx]
#         valid_stkcd = stkcd_matrix[:, valid_idx]
#         # 生成训练集
#         test_idx = (date_list >= int(args.test_start_date.replace('-', ''))) & (date_list <= int(args.test_end_date.replace('-', '')))
#         test_signal = signal_matrix[:, test_idx, :]
#         test_ret = return_matrix[:, test_idx]
#         test_mask = mask_matrix[:, test_idx]
#         test_date = date_matrix[:, test_idx]
#         test_stkcd = stkcd_matrix[:, test_idx]

#         train_data = (train_signal, train_ret, train_mask, train_date, train_stkcd)
#         valid_data = (valid_signal, valid_ret, valid_mask, valid_date, valid_stkcd)
#         test_data = (test_signal, test_ret, test_mask, test_date, test_stkcd)
#         data = (train_data, valid_data, test_data)
#         with open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/roll_{test_year}_balanced.pkl', 'wb') as file2:
#             pkl.dump(data, file2)


# def gen_fullSample_balance_matrix():
#     data_path = '/mnt/SSD4TB_1/data_output'
#     stock_data_path = f'/mnt/SSD4TB_1/data_output/data_exmkt_nontb_top2000_rank.h5'
#     signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')
#     to_f32 = {signal: 'float32' for signal in signal_list['signal'].values}

#     features = signal_list['signal'].values.tolist()
#     features.remove('C.3.8.1')
#     for year in range(2010, 2017):
#         print(year)

#         train_year = str(year) + '-01-01'
#         test_year = str(year+5) + '-12-31'
#         stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)

#         # get index df
#         train_data = stock_data[
#             (stock_data['date'] >= train_year) & (stock_data['date'] <= test_year)].copy()
#         train_data.set_index(['stkcd', 'date'], inplace=True)
#         index_levels = train_data.index.remove_unused_levels().levels
#         index = pd.MultiIndex.from_product(index_levels)
#         index_df = pd.DataFrame([], index=index)
#         del train_data


#         start_year = str(year - 2) + '-01-01'
#         end_year = test_year
#         est_data = stock_data[
#                 (stock_data['date'] >= start_year) & (stock_data['date'] <= end_year)].copy()

#         est_data.set_index(['stkcd', 'date'], inplace=True)
#         est_data = pd.merge(index_df, est_data, how='left', on=index.names)
#         train_array, train_mask, _, train_columns, date_matrix, stkcd_matrix = data_organization.convert_array(est_data)
#         del stock_data
#         signal_matrix, return_matrix, mask_matrix = split_signal_return_mask(train_array, train_columns, train_mask)
#         date_matrix = date_matrix.squeeze()
#         stkcd_matrix = np.expand_dims(stkcd_matrix, axis=1)
#         stkcd_matrix = np.tile(stkcd_matrix, (1, signal_matrix.shape[1]))
#         pkl.dump((signal_matrix, return_matrix, mask_matrix, date_matrix, stkcd_matrix), open(f'/mnt/HDD40TB/huahao/portfolio_construction/data/estMatrix_{year}_balanced.pkl', 'wb'),
#                  protocol=5)




# def divide_slice_train_valid_subsample():
#     data_path = '/mnt/SSD4TB_1/data_output'
#     signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')
#     to_f32 = {signal: 'float32' for signal in signal_list['signal'].values}
#     stock_data_path = f'/mnt/SSD4TB_1/data_output/data_exmkt_nontb_top2000_rank.h5'
#     stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)
#     # stkcd', 'date'
#     train_data = stock_data[
#         (stock_data['date'] >= '2015-01-01') & (stock_data['date'] <= '2015-06-30')]
#     train_data.set_index(['stkcd', 'date'], inplace=True)
#     train_array, train_mask, _, train_columns, train_date, train_stkcd = data_organization.convert_array(train_data)
#     del train_data
#     train_signal, train_return, train_mask = split_signal_return_mask(train_array, train_columns, train_mask)

#     valid_data = stock_data[
#         (stock_data['date'] >= '2015-07-01') & (stock_data['date'] <= '2015-09-30')]
#     valid_data.set_index(['stkcd', 'date'], inplace=True)
#     valid_array, valid_mask, _, valid_columns, valid_date, valid_stkcd = data_organization.convert_array(valid_data)
#     del valid_data
#     valid_signal, valid_return, valid_mask = split_signal_return_mask(valid_array, valid_columns, valid_mask)

#     test_data = stock_data[
#         (stock_data['date'] >= '2015-10-01') & (stock_data['date'] <= '2015-12-31')]
#     test_data.set_index(['stkcd', 'date'], inplace=True)
#     test_array, test_mask, _, test_columns, test_date, test_stkcd = data_organization.convert_array(test_data)
#     del test_data
#     test_signal, test_return, test_mask = split_signal_return_mask(test_array, test_columns, test_mask)

#     return (train_signal, train_return, train_mask, train_date, train_stkcd), \
#               (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd), \
#                 (test_signal, test_return, test_mask, test_date, test_stkcd)


# class ConstantMaskDataset(Dataset):
#     def __init__(self, data_input, args):
#         '''
#         :param data_input: (train_signal, train_return, train_mask, TDict_train, NDict_train,)
#         格式为[N, T, K]的矩阵
#         mask N * T
#         stkcd N
#         '''
#         self.args = args
#         self.signal_matrix = data_input[0]
#         self.return_matrix = data_input[1] * 100
#         self.mask_matrix = data_input[2]
#         self.time_matrix = torch.from_numpy(data_input[3])
#         self.stkcd_matrix = torch.from_numpy(data_input[4])

#     def __getitem__(self, index):
#         end = index + self.args.window_length
#         mask = self.mask_matrix[:, end - 1]
#         signal = self.signal_matrix[mask, index:end, :]
#         target = self.return_matrix[mask, end - 1: end - 1 + self.args.pred_length]
#         ret_series = self.return_matrix[mask, index:end-1]
#         time = self.time_matrix[mask, end - 1: end - 1 + self.args.pred_length, :]
#         stkcd = self.stkcd_matrix[mask]
#         print(mask.shape)
#         print(signal.shape)
#         print(target.shape)
#         print(ret_series.shape)
#         print(time.shape)
#         print(stkcd.shape)
#         if self.args.match_stkcd_date:
#             return signal, target, mask, ret_series, time, stkcd
#         else:
#             return signal, target, mask, ret_series

#     def __len__(self):
#         return self.signal_matrix.shape[1] - self.args.window_length + 2 - self.args.pred_length


# class RetDataset(Dataset):
#     def __init__(self, data_input, args):
#         '''
#         :param data_input: (train_signal, train_return, train_mask, TDict_train, NDict_train,)
#         格式为[N, T, K]的矩阵
#         mask N * T
#         stkcd N
#         '''
#         self.args = args
#         self.signal_matrix = data_input[0]
#         self.return_matrix = data_input[1] * 100
#         self.mask_matrix = data_input[2]
#         self.time_matrix = torch.from_numpy(data_input[3])
#         self.stkcd_matrix = torch.from_numpy(data_input[4])

#     def __getitem__(self, index):
#         index += 1
#         end = index + self.args.window_length
#         if self.args.match_stkcd_date:
#             return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args.pred_length],\
#                    self.mask_matrix[:, end - 1: end - 1 + self.args.pred_length], \
#                    self.return_matrix[:, index-1: end-1], self.time_matrix[:, end - 1: end - 1 + self.args.pred_length, :],\
#                    self.stkcd_matrix
#         else:
#             return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args.pred_length],\
#                    self.mask_matrix[:, end - 1: end - 1 + self.args.pred_length], \
#                    self.return_matrix[:, index-1: end-1]

#     def __len__(self):
#         return self.signal_matrix.shape[1] - self.args.window_length + 2 - self.args.pred_length - 1


class BalancedDataset(Dataset):
    def __init__(self, data_input, args, match_stkcd_date):
        '''
        :param data_input: (train_signal, train_return, train_lreturn, train_mask, TDict_train, NDict_train,)
        格式为[N, T, K]的矩阵
        mask N * T
        stkcd N * T
        '''
        self.args = args
        self.signal_matrix = data_input[0]
        self.return_matrix = data_input[1] * 100
        self.lreturn_matrix = data_input[2].astype('float32') * 100
        self.mask_matrix = data_input[3]
        self.time_matrix = torch.from_numpy(data_input[4])
        self.stkcd_matrix = torch.from_numpy(data_input[5])
        self.match_stkcd_date = match_stkcd_date

    def __getitem__(self, index):
        end = index + self.args['window_length']
        if self.match_stkcd_date:
            return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args['pred_length']],\
                self.mask_matrix[:, end - 1: end - 1 + self.args['pred_length']], \
                self.lreturn_matrix[:, index: end], self.time_matrix[:, end - 1: end - 1 + self.args['pred_length']],\
                self.stkcd_matrix[:, end - 1: end - 1 + self.args['pred_length']]
        else:
            return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args['pred_length']],\
                self.mask_matrix[:, end - 1: end - 1 + self.args['pred_length']], \
                self.lreturn_matrix[:, index: end]

    def __len__(self):
        return self.signal_matrix.shape[1] - self.args['window_length'] + 1 - self.args['pred_length'] + 1


class GraphDataset(Dataset):
    def __init__(self, data_input, args, adj_mat):
        '''
        :param data_input: (train_signal, train_return, train_mask, TDict_train, NDict_train,)
        格式为[N, T, K]的矩阵
        mask N * T
        stkcd N * T
        '''
        self.args = args
        self.signal_matrix = data_input[0]
        self.return_matrix = data_input[1] * 100
        self.mask_matrix = data_input[2]
        self.time_matrix = torch.from_numpy(data_input[3])
        self.stkcd_matrix = torch.from_numpy(data_input[4])
        self.adj_mat = adj_mat.type(torch.float32)

    def __getitem__(self, index):
        index += 1
        end = index + self.args.window_length
        if self.args.match_stkcd_date:
            return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args.pred_length],\
                   self.mask_matrix[:, end - 1: end - 1 + self.args.pred_length], \
                   self.return_matrix[:, index-1: end-1], self.time_matrix[:, end - 1: end - 1 + self.args.pred_length],\
                   self.stkcd_matrix[:, end - 1: end - 1 + self.args.pred_length],  self.adj_mat[end-1]
        else:
            return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args.pred_length],\
                   self.mask_matrix[:, end - 1: end - 1 + self.args.pred_length], \
                   self.return_matrix[:, index-1: end-1], self.adj_mat[end-1] 

    def __len__(self):
        return self.signal_matrix.shape[1] - self.args.window_length + 1 - self.args.pred_length


# class FactorDataset(Dataset):
#     def __init__(self, data_input, args):
#         '''
#         :param data_input: (train_signal, train_return, train_mask, TDict_train, NDict_train,)
#         格式为[N, T, K]的矩阵
#         mask N * T
#         stkcd N
#         '''
#         self.args = args
#         self.signal_matrix = data_input[0].float()
#         self.return_matrix = data_input[1].float() * 100
#         self.mask_matrix = data_input[2]
#         self.time_matrix = torch.from_numpy(data_input[3])
#         self.stkcd_matrix = torch.from_numpy(data_input[4])
#         self.factor_matrix = data_input[5].float()

#     def __getitem__(self, index):
#         index += 1
#         end = index + self.args.window_length
#         if self.args.nolead:
#             factor = self.factor_matrix[:, end-1: end-1+self.args.pred_length, 3:]
#         else:
#             factor = self.factor_matrix[:, end - 1: end - 1 + self.args.pred_length, :3]
#         if self.args.match_stkcd_date:
#             return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args.pred_length],\
#                    self.mask_matrix[:, end - 1: end - 1 + self.args.pred_length], \
#                    self.return_matrix[:, index-1: end-1], self.time_matrix[:, end - 1: end - 1 + self.args.pred_length, :],\
#                    self.stkcd_matrix, factor
#         else:
#             return self.signal_matrix[:, index:end, :], self.return_matrix[:, end - 1: end - 1 + self.args.pred_length],\
#                    self.mask_matrix[:, end - 1: end - 1 + self.args.pred_length], \
#                    self.return_matrix[:, index-1: end-1], factor

#     def __len__(self):
#         return self.signal_matrix.shape[1] - self.args.window_length + 2 - self.args.pred_length - 1


# def expand_balanced_panel(df, date_var='date', id_var='stkcd'):
#     all_trd_dates = np.sort(df[date_var].unique())
#     list_ticker = df[id_var].unique().tolist()
#     combination = list(itertools.product(all_trd_dates, list_ticker))
#     df_full = pd.DataFrame(combination, columns=[date_var, id_var]).merge(df, on=[date_var, id_var], how="left")
#     df_full = df_full.sort_values([date_var, id_var])
#     return df_full


# def select_subsample(args):
#     def split_signal_return_mask_array(df):
#         print(df.shape)
#         shape = (len(df['stkcd'].drop_duplicates()), len(df['date'].drop_duplicates()))
#         print(df.groupby('date')['mask'].sum())
#         print(df.groupby('date')['stkcd'].count())
#         print(shape)
#         mask = df['mask'].to_numpy().reshape(shape, order='F').astype('bool')
#         print(mask.shape)
#         print(np.sum(mask, axis=0))
#         # print(np.sum(comp_mask, axis=1))
#         df = df.drop(columns=['mask'])
#         array = df.to_numpy().reshape(shape + (-1,), order='F')
#         df['date'] = df['date'].apply(lambda x: x.year * 10000 + x.month * 100 + x.day).astype('int32')
#         date = df['date'].to_numpy().reshape(shape + (-1,), order='F')
#         stkcd = df['stkcd'].drop_duplicates().to_numpy().astype('int32')
#         input_CMat_t = np.nan_to_num(array[..., 4:], copy=False).astype(np.float32)
#         input_RMat_t = array[..., df.columns.get_loc('ret_open')].astype(np.float32)
#         input_CMat_t, input_RMat_t, input_IMat_t = torch.from_numpy(input_CMat_t), torch.from_numpy(input_RMat_t), torch.from_numpy(mask)
#         return input_CMat_t, input_RMat_t, input_IMat_t, date, stkcd

#     data_path = '/mnt/SSD4TB_1/data_output'
#     signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')
#     data_path = '/mnt/SSD4TB_1/data_output'
#     if args.mktcap_filter == 2000:
#         stock_data_path = f'{data_path}/data_exmkt_nontb_top2000_norm.h5'
#     else:
#         raise NotImplementedError
#     signal_list = pd.read_pickle(data_path + '/' + 'signal_list.pkl')
#     to_f32 = {signal: 'float32' for signal in signal_list['signal'].values}
#     stock_data = pd.read_hdf(stock_data_path, dtype=to_f32)
#     # stkcd', 'date'
#     hs300_comp = pd.read_pickle('/mnt/HDD16TB/huahao/portfolio_construction/data/HS300_component.pkl')
#     hs300_comp = hs300_comp.rename(columns={'Enddt': 'date', 'Stkcd': 'stkcd'})
#     hs300_comp['mask'] = 1
#     hs300_comp = hs300_comp.fillna(0)
#     grouped = hs300_comp.groupby('date')['mask'].sum()
#     stkcd_list = hs300_comp['stkcd'].unique()
#     stock_data = stock_data[stock_data['stkcd'].isin(stkcd_list)]
#     df_filled = expand_balanced_panel(stock_data)
#     df_filled = pd.merge(df_filled, hs300_comp[['date', 'stkcd', 'mask']], on=['stkcd', 'date'], how='left')
#     df_filled = df_filled.fillna(0)

#     train_data = df_filled[
#         (df_filled['date'] >= args.train_start_date) & (df_filled['date'] <= args.train_end_date)].copy()
#     print(train_data.groupby('date')['mask'].sum())
#     train_signal, train_return, train_mask, train_date, train_stkcd = split_signal_return_mask_array(train_data)

#     valid_data = df_filled[
#         (df_filled['date'] >= args.valid_start_date) & (df_filled['date'] <= args.valid_end_date)].copy()
#     valid_signal, valid_return, valid_mask, valid_date, valid_stkcd = split_signal_return_mask_array(valid_data)

#     test_data = df_filled[
#         (df_filled['date'] >= args.test_start_date) & (df_filled['date'] <= args.test_end_date)].copy()
#     test_signal, test_return, test_mask, test_date, test_stkcd = split_signal_return_mask_array(test_data)

#     train = (train_signal, train_return, train_mask, train_date, train_stkcd)
#     valid = (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd)
#     test = (test_signal, test_return, test_mask, test_date, test_stkcd)

#     test_year = args.test_start_date.split('-')[0]
#     pkl.dump((train, valid, test), open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/hs300_{test_year}.pkl', 'wb'))

#     return (train_signal, train_return, train_mask, train_date, train_stkcd), \
#            (valid_signal, valid_return, valid_mask, valid_date, valid_stkcd), \
#            (test_signal, test_return, test_mask, test_date, test_stkcd)



# def main():
#     select_subsample()


# if __name__ == '__main__':
#     main()