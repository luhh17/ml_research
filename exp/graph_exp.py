import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN
from models.transformer.transformer import Transformer
from models.gnn.gcn import GCN, GCN_with_encoder, GCN_2_encoder, GCN_before_encoder
from models.gnn.gat import GAT_with_encoder
from models.gnn.dual import Dual
from models.gnn.sem_graph import Sem_Graph, Sem_Graph_v2
from models.ffn import MLP_linear, MLP_ts, MLP_post_ts, MLP_RNN
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import RetDataset, BalancedDataset, GraphDataset
from utils.loss_func import RetMSELoss
from utils.normalize import ts_mean_var_norm, ts_min_max_norm
from utils.graph_matrix import norm_adj, dummy_adj
import joblib
import pdb
'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class GraphExp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_data()
        self.model = self._build_model().to(self.device)
        # dist.init_process_group(backend='nccl', init_method='env://')
        if self.args.est_method == 'rolling':
            self.save_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_checkpoint.pt"
        else:
            self.save_path = f'{self.args.model_file_path}/checkpoint.pt'
        self.model = self.model.to(self.device)

    def save_args(self):
        if self.args.est_method == 'rolling':
            pkl.dump(self.args, open(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_config.pt", 'wb'))
        else:
            pkl.dump(self.args, open(f'{self.args.model_file_path}/config.pt', 'wb'))

    def _build_model(self):
        if self.args.model == 'rnn':
            model = RNN(self.args)
        elif self.args.model == 'transformer':
            model = Transformer(self.args)
        elif self.args.model == 'informer':
            model = Informer(self.args)
        elif self.args.model == 'gat_encoder':
            model = GAT_with_encoder(self.args)
        elif self.args.model == 'mlp':
            model = MLP_linear(self.args, out_dim=self.args.pred_length)
        elif self.args.model == 'gcn':
            model = GCN(self.args)
        elif self.args.model == 'sem_graph':
            model = Sem_Graph(self.args)
        elif self.args.model == 'sem_graph_v2':
            model = Sem_Graph_v2(self.args)
        elif self.args.model == 'gcn_encoder':
            model = GCN_with_encoder(self.args)
        elif self.args.model == 'gcn_2_encoder':
            model = GCN_2_encoder(self.args)
        elif self.args.model == 'gcn_before_encoder':
            model = GCN_before_encoder(self.args)
        elif self.args.model == 'dual':
            model = Dual(self.args)

        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _acquire_device(self):
        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]

        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def cal_daily_matrix(self, year, train_num, valid_num, test_num):
        if os.path.exists(f'../data/cross_matrix_{year}_{self.args.corr_time}_{self.args.est_window}'):
            print('load cross matrix')
            cross_matrix = pkl.load(
                open(f'../data/cross_matrix_{year}_{self.args.corr_time}_{self.args.est_window}', 'rb'))
        else:
            print('calculate cross matrix')
            signal_matrix, return_matrix, mask_matrix, date_matrix, stkcd_matrix = pkl.load(
                open(f'../data/estMatrix_{year - 5}_balanced.pkl', 'rb'))
            true_start = signal_matrix.shape[1] - train_num - valid_num - test_num
            cross_matrix_list = []
            
            for t in range(true_start, signal_matrix.shape[1]):
                # return 应该再提前一天，true_start 当天的收益率其实在true_start - 1
                if self.args.corr_time == 'cont':
                    past_ret = return_matrix[:, t - self.args.est_window: t]
                    cross_matrix = torch.matmul(past_ret, past_ret.T) / (self.args.est_window)
                elif self.args.corr_time == 'lag':
                    ret_series = return_matrix[:, t - self.args.est_window: t]
                    past_ret = return_matrix[:, t - self.args.est_window - 1: t - 1]
                    # N x N, (i, j) represent the correlation between stock i and stock j's last day's return
                    cross_matrix = torch.matmul(ret_series, past_ret.T) / (self.args.est_window)
                cross_matrix_list.append(cross_matrix)
            cross_matrix = torch.stack(cross_matrix_list, dim=0)
            cross_matrix = cross_matrix.half()
            print('success convert')
            joblib.dump(cross_matrix, open(f'../data/cross_matrix_{year}_{self.args.corr_time}_{self.args.est_window}', 'wb'))
        # T x N x N
        train_cross = cross_matrix[:train_num]
        valid_cross = cross_matrix[train_num: train_num + valid_num]
        test_cross = cross_matrix[train_num + valid_num:]
        return train_cross, valid_cross, test_cross

    def divide_train_valid_test(self, train_data, valid_data, test_data):

        test_start_idx = train_data.shape[1] + valid_data.shape[1]
        valid_start_idx = train_data.shape[1]
        new_data = np.concatenate([train_data, valid_data, test_data], axis=1)

        new_train = new_data[:, :valid_start_idx - self.args.window_length + 1]
        new_valid = new_data[:,
                    valid_start_idx - self.args.window_length + 1: test_start_idx - self.args.window_length + 1]
        new_test = new_data[:, test_start_idx - self.args.window_length + 1:]

        return new_train, new_valid, new_test

    def _get_data(self):
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}_balanced.pkl', 'rb'))

        # N x T x C
        train_signal, train_ret, train_mask, train_time, train_stkcd = train
        valid_signal, valid_ret, valid_mask, valid_time, valid_stkcd = valid
        test_signal, test_ret, test_mask, test_time, test_stkcd = test

        train_signal, valid_signal, test_signal = self.divide_train_valid_test(train_signal, valid_signal, test_signal)
        train_num = train_signal.shape[1]
        valid_num = valid_signal.shape[1]
        test_num = test_signal.shape[1]
        if self.args.model != 'mlp' and self.args.model != 'sem_graph' and self.args.model != 'sem_graph_v2':
            train_cross, valid_cross, test_cross = self.cal_daily_matrix(int(self.args.test_start_date.split('-')[0]),
                                                                         train_num, valid_num, test_num)
        elif self.args.model == 'sem_graph' or self.args.model == 'sem_graph_v2':
            train_cross, valid_cross, test_cross = None, None, None

        train_ret, valid_ret, test_ret = self.divide_train_valid_test(train_ret, valid_ret, test_ret)
        train_mask, valid_mask, test_mask = self.divide_train_valid_test(train_mask, valid_mask, test_mask)
        train_time, valid_time, test_time = self.divide_train_valid_test(train_time, valid_time, test_time)
        train_stkcd, valid_stkcd, test_stkcd = self.divide_train_valid_test(train_stkcd, valid_stkcd, test_stkcd)
        train = (train_signal, train_ret, train_mask, train_time, train_stkcd)
        valid = (valid_signal, valid_ret, valid_mask, valid_time, valid_stkcd)
        test = (test_signal, test_ret, test_mask, test_time, test_stkcd)

        if self.args.model == 'sem_graph' or self.args.model == 'sem_graph_v2':
            train_dataset = BalancedDataset(train, self.args)
            valid_dataset = BalancedDataset(valid, self.args)
            test_dataset = BalancedDataset(test, self.args)
        else:
            train_dataset = GraphDataset(train, self.args, train_cross)
            valid_dataset = GraphDataset(valid, self.args, valid_cross)
            test_dataset = GraphDataset(test, self.args, test_cross)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

    def process_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series, adj_mat) in enumerate(dataloader):
            mask = mask.reshape(-1, self.args.pred_length)
            signal = signal.to(self.device).squeeze(2)
            ret_series = ret_series[:, :, -1:].to(self.device)
            ret = ret.reshape(-1, self.args.pred_length).to(self.device)
            mask = mask.to(self.device)
            if self.args.only_ret:
                signal = ret_series
            if self.args.weighted:
                adj_mat = norm_adj(adj_mat, self.args.add_eye)
            else:
                adj_mat = dummy_adj(adj_mat, self.args.weight_threshold, self.args.add_eye)
            adj_mat = adj_mat.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                if self.args.model == 'mlp':
                    signal = signal.reshape(-1, self.args.window_length, self.args.char_dim)
                    preds = self.model(signal, y_enc=ret_series)
                else:
                    preds = self.model(signal, adj_mat)
                preds = preds.reshape(-1, self.args.pred_length)
                preds = preds * mask
                ret = ret * mask
                loss = criterion(preds, ret)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def train(self):
        train_steps = len(self.train_loader)
        self.args.match_stkcd_date = False
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.2, patience=4, verbose=True)
        criterion = torch.nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.epochs):
            iter_count = 0
            epoch_time = time.time()
            '''
            signal: (batch_size, num_stock, seq_len, feature_dim)
            ret_series: (batch_size, num_stock, seq_len, 1)
            mask: (batch_size, num_stock, seq_len)
            '''
            iter_count += 1
            self.model.train()
            if self.args.model == 'sem_graph' or self.args.model == 'sem_graph_v2':
                train_loss = self.cross_stock_atten_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            elif self.args.model == 'dual':
                train_loss = self.dual_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            else:
                train_loss = self.process_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            self.model.eval()
            with torch.no_grad():
                if self.args.model == 'sem_graph' or self.args.model == 'sem_graph_v2':
                    vali_loss = self.cross_stock_atten_epoch(self.valid_loader, criterion, train=False)
                elif self.args.model == 'dual':
                    vali_loss = self.dual_epoch(self.valid_loader, criterion, train=False)
                else:
                    vali_loss = self.process_one_epoch(self.valid_loader, criterion, train=False)
            scheduler.step(vali_loss)
            print("Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss, model_optim.state_dict()['param_groups'][0]['lr']))
            # torch.save(self.model.state_dict(), self.save_path)
            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.model.load_state_dict(torch.load(self.save_path))
        return self.model


    def test_model(self):
        self.args.match_stkcd_date = True
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        df_list = []
        with torch.no_grad():
            for i, (data) in enumerate(self.test_loader):
                signal = data[0]
                mask = data[2].reshape(-1, self.args.pred_length)
                atten_mask = data[2].type(torch.float)
                atten_mask = torch.bmm(atten_mask, atten_mask.transpose(1, 2)).type(torch.bool).to(self.device)
                signal = signal.to(self.device).squeeze(2)
                ret_series = data[3][:, :, -1:].to(self.device)
                if self.args.model != 'sem_graph' and self.args.model != 'sem_graph_v2':
                    if self.args.weighted:
                        adj = norm_adj(data[6], self.args.add_eye).to(self.device)
                        # exit()
                    else:
                        # print(self.args.weight_threshold)
                        adj = dummy_adj(data[6], self.args.weight_threshold, self.args.add_eye).to(self.device)
                if self.args.only_ret:
                    signal = ret_series
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                    if self.args.model == 'mlp':
                        signal = signal.reshape(-1, self.args.window_length, self.args.char_dim)
                        preds = self.model(signal, y_enc=ret_series)
                    elif self.args.model == 'sem_graph' or self.args.model == 'sem_graph_v2':
                        preds, adj_mat = self.model(signal, ret_series, atten_mask=atten_mask)
                        preds = preds.reshape(-1, self.args.pred_length)
                    elif self.args.model == 'dual':
                        preds, adj_mat = self.model(signal, y_enc=ret_series, atten_mask=atten_mask, syn_adj=adj)
                        preds = preds.reshape(-1, self.args.pred_length)
                    else:
                        preds = self.model(signal, adj)
                preds = preds.detach().cpu().numpy().reshape(-1, self.args.pred_length)
                ret = data[1].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                mask = data[2].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                if self.args.match_stkcd_date:
                    time = data[4].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                    stkcd = data[5].detach().cpu().numpy().reshape(-1)
                    df_dict = {'stkcd': stkcd, 'date': time[:, 0]}
                    df_dict[f'ret'] = ret.squeeze()
                    df_dict[f'pred'] = preds.squeeze()
                    df_dict[f'mask'] = mask.squeeze()
                    df = pd.DataFrame(df_dict)
                    df_list.append(df)
        df = pd.concat(df_list)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result',)


    def ortho_regu(self, adj_mat, atten_mask):
        # adj_mat B x N x N
        regu_loss = torch.linalg.matrix_norm((torch.matmul(adj_mat, adj_mat.transpose(1, 2)) - torch.ones_like(adj_mat)) * atten_mask, ord='fro')
        return torch.mean(regu_loss)


    def cross_stock_atten_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series) in enumerate(dataloader):
            B, N, T, C = signal.shape
            B, N, T = mask.shape
            # print(mask.shape, ret_series.shape, ret.shape, signal.shape)
            mask = mask.squeeze(2)
            signal = signal.to(self.device).squeeze(2)
            ret_series = ret_series[:, :, -1:].to(self.device)
            ret = ret.squeeze(2).to(self.device)
            mask = mask.to(self.device)
            atten_mask = mask.unsqueeze(-1).type(torch.float)
            atten_mask = torch.bmm(atten_mask, atten_mask.transpose(1, 2)).type(torch.bool)

            if self.args.only_ret:
                signal = ret_series
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                preds, adj_mat = self.model(signal, y_enc=ret_series, atten_mask=atten_mask)
                # print(torch.sum(torch.isnan(preds)), torch.sum(torch.isnan(adj_mat)), torch.sum(torch.isnan(atten_mask)))
                preds = preds.reshape(B, N)
                # print(preds.shape, ret.shape, mask.shape, adj_mat.shape)
                preds = preds * mask
                ret = ret * mask
                loss = criterion(preds, ret)
                regu_loss = self.ortho_regu(adj_mat, atten_mask) * self.args.regu_weight
                # print('mse loss', loss.item(), 'regu loss', regu_loss.item())
                loss += regu_loss
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def dual_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series, syn_adj) in enumerate(dataloader):
            B, N, T, C = signal.shape
            B, N, T = mask.shape
            # print(mask.shape, ret_series.shape, ret.shape, signal.shape)
            mask = mask.squeeze(2)
            signal = signal.to(self.device).squeeze(2)
            ret_series = ret_series[:, :, -1:].to(self.device)
            ret = ret.squeeze(2).to(self.device)
            mask = mask.to(self.device)
            atten_mask = mask.unsqueeze(-1).type(torch.float)
            atten_mask = torch.bmm(atten_mask, atten_mask.transpose(1, 2)).type(torch.bool)
            if self.args.weighted:
                # print(syn_adj.shape)
                syn_adj = norm_adj(syn_adj, self.args.add_eye)
                # exit()
            else:
                # print(syn_adj.shape, signal.shape, ret.shape)
                syn_adj = dummy_adj(syn_adj, self.args.weight_threshold, self.args.add_eye)
            syn_adj = syn_adj.to(self.device)
            if self.args.only_ret:
                signal = ret_series
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                preds, adj_mat = self.model(signal, syn_adj=syn_adj, y_enc=ret_series, atten_mask=atten_mask)
                # print(torch.sum(torch.isnan(preds)), torch.sum(torch.isnan(adj_mat)), torch.sum(torch.isnan(atten_mask)))
                preds = preds.reshape(B, N)
                # print(preds.shape, ret.shape, mask.shape, adj_mat.shape)
                preds = preds * mask
                ret = ret * mask
                loss = criterion(preds, ret)
                regu_loss = self.ortho_regu(adj_mat, atten_mask) * self.args.regu_weight
                # print('mse loss', loss.item(), 'regu loss', regu_loss.item())
                loss += regu_loss
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss
