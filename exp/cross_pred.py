import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_selection import f_regression, mutual_info_regression
from models.rnn import RNN
from models.transformer import Transformer, Informer
from models.ffn import MLP_linear, MLP_ts, MLP_post_ts, MLP_RNN
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import RetDataset, divide_train_valid_test, BalancedDataset
from utils.loss_func import RetMSELoss
from utils.normalize import ts_mean_var_norm, ts_min_max_norm
'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class Cross_Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_data()
        if self.args.est_method == 'rolling':
            self.save_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_checkpoint.pt"
        else:
            self.save_path = f'{self.args.model_file_path}/checkpoint.pt'

    def save_args(self):
        if self.args.est_method == 'rolling':
            pkl.dump(self.args, open(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_config.pt", 'wb'))
        else:
            pkl.dump(self.args, open(f'{self.args.model_file_path}/config.pt', 'wb'))

    def _acquire_device(self):
        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]
        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def _get_data(self):
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}_balanced.pkl', 'rb'))
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        # N x T x C

        train_dataset = BalancedDataset(train, self.args)
        valid_dataset = BalancedDataset(valid, self.args)
        test_dataset = BalancedDataset(test, self.args)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)


    def cal_mi_matrix(self, signal_matrix, target_matrix):
        N, T = signal_matrix.shape
        signal_matrix = signal_matrix.transpose(1, 0)
        # mi_matrix 第一行代表N个特征与第一个股票的关系
        mi_matrix = np.zeros((N, N))
        for i in range(N):
            target = target_matrix[i]
            mi = mutual_info_regression(signal_matrix, target)
            mi_matrix[i] = mi
        return mi_matrix


    def process_one_epoch(self, data):
        signal_matrix, return_matrix, mask_matrix, time_matrix, stkcd_matrix = data
        N, T, C = signal_matrix.shape
        df_list = []
        cross_list = []
        for t in range(self.args.window_length+1, T):
            future_ret = return_matrix[:, t]
            ret = return_matrix[:, t-self.args.window_length:t]
            if self.args.signal == 'ret':
                signal_series = return_matrix[:, t-self.args.window_length-1:t-1]
                signal = return_matrix[:, t - 1].reshape(-1, 1)
            elif self.args.signal == 'tvr':
                signal_list = pd.read_pickle('/mnt/SSD4TB_1/data_output' + '/' + 'signal_list.pkl')
                features = signal_list['signal'].values
                tvr_idx = np.where(features == 'A.5.1.1')[0][0]
                signal_series = signal_matrix[:, t - self.args.window_length - 1:t - 1, tvr_idx]
                signal = signal_matrix[:, t - 1, tvr_idx].reshape(-1, 1)
            elif self.args.signal == 'volume':
                signal_list = pd.read_pickle('/mnt/SSD4TB_1/data_output' + '/' + 'signal_list.pkl')
                features = signal_list['signal'].values
                tvr_idx = np.where(features == 'A.5.2.1')[0][0]
                signal_series = signal_matrix[:, t - self.args.window_length - 1:t - 1, tvr_idx]
                signal = signal_matrix[:, t - 1, tvr_idx].reshape(-1, 1)
            elif self.args.signal == 'liquid':
                signal_list = pd.read_pickle('/mnt/SSD4TB_1/data_output' + '/' + 'signal_list.pkl')
                features = signal_list['signal'].values
                tvr_idx = np.where(features == 'A.5.3.1')[0][0]
                signal_series = signal_matrix[:, t - self.args.window_length - 1:t - 1, tvr_idx]
                signal = signal_matrix[:, t - 1, tvr_idx].reshape(-1, 1)
            if self.args.cross_relation == 'linear':
                cross_matrix = np.matmul(ret, signal_series.T) / (self.args.window_length)
            elif self.args.cross_relation == 'mi':
                cross_matrix = self.cal_mi_matrix(signal_series, ret)
            u, s, vh = scipy.sparse.linalg.svds(cross_matrix, which='LM', k=self.args.topk)
            S = np.diag(s)
            L = np.matmul(vh.T, u.T)

            weight = np.matmul(signal.T, L)
            time_tag = time_matrix[:, t].squeeze()
            stkcd = stkcd_matrix[:, t].squeeze()
            mask = mask_matrix[:, t].squeeze()
            df_dict = {'stkcd': stkcd, 'date': time_tag}
            df_dict[f'ret'] = future_ret.squeeze()
            df_dict[f'pred'] = weight.squeeze()
            df_dict[f'mask'] = mask.squeeze()
            df = pd.DataFrame(df_dict)
            df_list.append(df)
            print(time_tag[0])
        df = pd.concat(df_list)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result', )

    def enhance_signal(self, data):
        signal_matrix, return_matrix, mask_matrix, time_matrix, stkcd_matrix = data
        N, T, C = signal_matrix.shape
        df_list = []
        cross_list = []
        for t in range(self.args.window_length + 1, T):
            future_ret = return_matrix[:, t]
            ret = return_matrix[:, t - self.args.window_length:t]
            if self.args.signal == 'ret':
                signal_series = return_matrix[:, t - self.args.window_length - 1:t - 1]
                signal = return_matrix[:, t - 1].reshape(-1, 1)
            elif self.args.signal == 'tvr':
                signal_list = pd.read_pickle('/mnt/SSD4TB_1/data_output' + '/' + 'signal_list.pkl')
                features = signal_list['signal'].values
                tvr_idx = np.where(features == 'A.5.1.1')[0][0]
                signal_series = signal_matrix[:, t - self.args.window_length - 1:t - 1, tvr_idx]
                signal = signal_matrix[:, t - 1, tvr_idx].reshape(-1, 1)
            elif self.args.signal == 'volume':
                signal_list = pd.read_pickle('/mnt/SSD4TB_1/data_output' + '/' + 'signal_list.pkl')
                features = signal_list['signal'].values
                tvr_idx = np.where(features == 'A.5.2.1')[0][0]
                signal_series = -signal_matrix[:, t - self.args.window_length - 1:t - 1, tvr_idx]
                signal = -signal_matrix[:, t - 1, tvr_idx].reshape(-1, 1)
            elif self.args.signal == 'liquid':
                signal_list = pd.read_pickle('/mnt/SSD4TB_1/data_output' + '/' + 'signal_list.pkl')
                features = signal_list['signal'].values
                tvr_idx = np.where(features == 'A.5.3.1')[0][0]
                signal_series = signal_matrix[:, t - self.args.window_length - 1:t - 1, tvr_idx]
                signal = signal_matrix[:, t - 1, tvr_idx].reshape(-1, 1)
            if self.args.cross_relation == 'linear':
                cross_matrix = np.matmul(ret, signal_series.T) / (self.args.window_length)
            elif self.args.cross_relation == 'mi':
                cross_matrix = self.cal_mi_matrix(signal_series, ret)
            elif self.args.cross_relation == 'self':
                cross_matrix = np.matmul(signal_series, signal_series.T) / (self.args.window_length)
            if self.args.enhance == 'raw':
                weight = signal
            elif self.args.enhance == 'enhance_raw':
                denoise_matrix = cross_matrix
                enhanced_signal = denoise_matrix @ signal
                if self.args.normalize == 'abs':
                    enhanced_signal = enhanced_signal.squeeze() / np.sum(np.abs(denoise_matrix), axis=1)
                elif self.args.normalize == 'sum':
                    enhanced_signal = enhanced_signal.squeeze() / np.sum(denoise_matrix, axis=1)
                weight = enhanced_signal
            elif self.args.enhance == 'enhance_svd':
                u, s, vh = scipy.sparse.linalg.svds(cross_matrix, which='LM', k=self.args.topk)
                denoise_matrix = u @ np.diag(s) @ vh
                # denoise_matrix = cross_matrix
                enhanced_signal = denoise_matrix @ signal
                if self.args.normalize == 'abs':
                    enhanced_signal = enhanced_signal.squeeze() / np.sum(np.abs(denoise_matrix), axis=1)
                elif self.args.normalize == 'sum':
                    enhanced_signal = enhanced_signal.squeeze() / np.sum(denoise_matrix, axis=1)
                weight = enhanced_signal
            elif self.args.enhance == 'pp':
                u, s, vh = scipy.sparse.linalg.svds(cross_matrix, which='LM', k=self.args.topk)
                denoise_matrix = u @ vh
                # denoise_matrix = cross_matrix
                enhanced_signal = denoise_matrix @ signal
                if self.args.normalize == 'abs':
                    enhanced_signal = enhanced_signal.squeeze() / np.sum(np.abs(denoise_matrix), axis=1)
                elif self.args.normalize == 'sum':
                    enhanced_signal = enhanced_signal.squeeze() / np.sum(denoise_matrix, axis=1)
                weight = enhanced_signal
            # weight = np.matmul(signal.T, L)
            time_tag = time_matrix[:, t].squeeze()
            stkcd = stkcd_matrix[:, t].squeeze()
            mask = mask_matrix[:, t].squeeze()
            df_dict = {'stkcd': stkcd, 'date': time_tag}
            df_dict[f'ret'] = future_ret.squeeze()
            df_dict[f'pred'] = weight.squeeze()
            df_dict[f'mask'] = mask.squeeze()
            df = pd.DataFrame(df_dict)
            df_list.append(df)
            print(time_tag[0])
        df = pd.concat(df_list)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result', )

    def train(self):
        train_signal, train_ret, train_mask, train_time, train_stkcd = self.train_data
        valid_signal, valid_ret, valid_mask, valid_time, valid_stkcd = self.valid_data
        test_signal, test_ret, test_mask, test_time, test_stkcd = self.test_data
        test_start_idx = train_signal.shape[1] + valid_signal.shape[1]
        signal_matrix = np.concatenate([train_signal, valid_signal, test_signal], axis=1)[:, test_start_idx - self.args.window_length + 1:, :]
        return_matrix = np.concatenate([train_ret, valid_ret, test_ret], axis=1)[:, test_start_idx - self.args.window_length + 1:]
        mask_matrix = np.concatenate([train_mask, valid_mask, test_mask], axis=1)[:, test_start_idx - self.args.window_length + 1:]
        time_matrix = np.concatenate([train_time, valid_time, test_time], axis=1)[:, test_start_idx - self.args.window_length + 1:]
        stkcd_matrix = np.concatenate([train_stkcd, valid_stkcd, test_stkcd], axis=1)[:, test_start_idx - self.args.window_length + 1:]
        if self.args.enhance == 'pp':
            self.process_one_epoch([signal_matrix, return_matrix, mask_matrix, time_matrix, stkcd_matrix])
        else:
            self.enhance_signal([signal_matrix, return_matrix, mask_matrix, time_matrix, stkcd_matrix])

