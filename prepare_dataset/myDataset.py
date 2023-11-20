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

