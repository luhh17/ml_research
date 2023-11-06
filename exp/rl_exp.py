import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN
from models.transformer.transformer import Transformer
from models.targetformer.targetformer import Targetformer
from models.ffn import MLP_conv, MLP_linear, serial_MLP, Diff_gap_model
from models.tcn import TemporalConvNet
from models.dlinear import TLinear
from utils.masking import generate_lookback_mask
import torch.distributed as dist
import torch_package.functions as functions
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import copy
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import RetDataset
from utils.loss_func import RetMSELoss
'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


'''
Update on 2023.7
1. 增加了针对balanced数据集的支持，每一次训练，在train，valid，test上的截面股票个数相同
2. mode='normal'时，train函数只能处理一天的输入，对于transformer多天的回看窗口，需要重新设置
应该只保留两个模式：serial和two steps
3. 增加了对transformer等变体时间序列模型的支持
'''


class RL_Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_data_balanced()
        self.model = self._build_model().to(self.device)
        if self.args.est_method == 'rolling':
            self.save_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_checkpoint.pt"
        else:
            self.save_path = f'{self.args.model_file_path}/checkpoint.pt'

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
        # elif self.args.model == 'informer':
        #     model = SerialIterModel(self.args, Informer(self.args).to(self.device))
        # elif self.args.model == 'tcn':
        #     model = SerialIterModel(self.args, TemporalConvNet(self.args).to(self.device))
        elif self.args.model == 'targetformer':
            model = Targetformer(self.args, all_steps=True)
        elif self.args.model == 'mlp':
            if self.args.mlp_implement == 'linear':
                model = MLP_linear(self.args, out_dim=1)
            elif self.args.mlp_implement == 'conv':
                model = MLP_conv(self.args)
        # elif self.args.model == 'tlinear':
        #     model = SerialIterModel(self.args, TLinear(self.args).to(self.device))
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model)
        return model

    def _acquire_device(self):

        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]

        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def _get_data(self):
        # train, valid, test = divide_train_valid_test(self.args)
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}.pkl', 'rb'))
        # train, valid, test = divide_slice_train_valid_test()
        # 原始矩阵 N x T x C or N x T
        # transpose, 得到T x N x C
        train = [np.transpose(t, (1, 0, 2)) if len(t.shape) == 3 else np.transpose(t, (1, 0)) for t in train[:3]]
        valid = [np.transpose(t, (1, 0, 2)) if len(t.shape) == 3 else np.transpose(t, (1, 0)) for t in valid[:3]]
        test = [np.transpose(t, (1, 0, 2)) if len(t.shape) == 3 else np.transpose(t, (1, 0)) for t in test[:3]]
        self.train_data = train
        self.valid_data = valid
        self.test_data = test


    def _select_optimizer(self):
        if self.args.optim == 'adam':
            return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

    '''
    第一步，根据当前的policy，采样batch条episode
    '''
    def prob_to_weight(self, scores, mask, device):
        # scores 输入应当为 T x N
        # mask 输入应当为 T x N
        g_num = self.args.topk
        prob = torch.from_numpy(np.zeros((scores.shape[0], scores.shape[1]))).half().to(device)
        weights = torch.from_numpy(np.zeros((scores.shape[0], scores.shape[1]))).half().to(device)
        # long_scores = scores * mask
        long_s, long_idx = torch.topk(scores, g_num, dim=1)
        # long_ratio = torch.softmax(long_s, dim=1)
        # short_scores = (1 - scores) * mask
        short_s, short_idx = torch.topk(scores, g_num, dim=1, largest=False)
        # short_ratio = torch.softmax(short_s, dim=1)
        for i, indice in enumerate(long_idx):
            weights[i, indice] = long_s[i]
            prob[i, indice] = long_s[i]
        for i, indice in enumerate(short_idx):
            weights[i, indice] = short_s[i]
            prob[i, indice] = short_s[i]
        prob = prob.contiguous()
        weights = weights.contiguous()
        return prob, weights

    def get_batch_data(self, signal_matrix, ret_matrix, mask_matrix, batch_start_idx):
        batch_start_idx = torch.from_numpy(batch_start_idx)
        signal_start_idx = torch.stack([batch_start_idx-i for i in range(self.args.window_length-1, -1, -1)], dim=1)
        batch_ret = ret_matrix[batch_start_idx]
        # batch x num_stock
        batch_ret = batch_ret.to(self.device)
        # batch x num_stock x seq_len x feature
        batch_data = signal_matrix[signal_start_idx]
        batch_data = batch_data.permute(0, 2, 1, 3).reshape(-1, self.args.window_length, self.args.char_dim).to(self.device)

        batch_mask = mask_matrix[batch_start_idx]
        batch_mask = batch_mask.to(self.device)
        return batch_data, batch_ret, batch_mask


    def sample_step_by_step(self, signal_matrix, ret_matrix, mask_matrix, optimizer, policy_net, train=True):
        # 原始矩阵 N x T x C, transpose 0, 1, 得到T x N x C
        signal_matrix = signal_matrix.transpose(0, 1)
        ret_matrix = ret_matrix.transpose(0, 1)
        mask_matrix = mask_matrix.transpose(0, 1)
        start_idx = np.arange(self.args.window_length-1, signal_matrix.shape[0] - self.args.max_steps + 1)
        np.random.shuffle(start_idx)
        batch_num = (signal_matrix.shape[0] - self.args.max_steps - self.args.window_length + 1) // self.args.batch_size + 1
        epoch_ret = []
        epoch_loss = []
        for batch_idx in range(batch_num):
            ret_list = []
            with torch.no_grad():
                for step_idx in range(self.args.max_steps):
                    batch_start_idx = start_idx[batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] + step_idx
                    batch_data, batch_ret, batch_mask = self.get_batch_data(signal_matrix, ret_matrix, mask_matrix, batch_start_idx)
                    scores = torch.sigmoid(policy_net(batch_data)).detach().reshape(batch_start_idx.shape[0], -1)
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                    cross_ret = torch.sum(batch_ret * weight, dim=1).detach()
                    ret_list.append(cross_ret)
            ret_list = torch.stack(ret_list, dim=1)
            epoch_ret.append(ret_list)
            if train:
                for step_idx in range(self.args.max_steps):
                    batch_start_idx = start_idx[batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] + step_idx
                    batch_data, batch_ret, batch_mask = self.get_batch_data(signal_matrix, ret_matrix, mask_matrix,
                                                                            batch_start_idx)
                    scores = torch.sigmoid(policy_net(batch_data)).reshape(batch_start_idx.shape[0], -1)
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                    if self.args.target == 'sharpe':
                        gt = torch.mean(ret_list, dim=1) / torch.std(ret_list, dim=1)
                        # gt = (gt - torch.mean(gt)) / (torch.std(gt) + 1e-9)
                    elif self.args.target == 'logret':
                        log_ret = torch.log(ret_list + 1)
                        gt = 0
                        for i in range(step_idx, ret_list.shape[1]):
                            gt += log_ret[:, i] * (self.args.gamma ** (i - step_idx))
                        # gt = (gt - torch.mean(gt)) / (torch.std(gt) + 1e-9)
                    gt = gt.reshape(-1, 1)
                    loss = -torch.mean(gt * torch.log(torch.softmax(prob, dim=1)))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
            else:
                with torch.no_grad():
                    for step_idx in range(self.args.max_steps):
                        batch_start_idx = start_idx[batch_idx * self.args.batch_size: (batch_idx + 1) * self.args.batch_size] + step_idx
                        batch_data, batch_ret, batch_mask = self.get_batch_data(signal_matrix, ret_matrix, mask_matrix,
                                                                                batch_start_idx)
                        scores = torch.sigmoid(policy_net(batch_data)).detach().reshape(batch_start_idx.shape[0], -1)
                        prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                        if self.args.target == 'sharpe':
                            gt = torch.mean(ret_list, dim=1) / torch.std(ret_list, dim=1)
                            # gt = (gt - torch.mean(gt)) / (torch.std(gt) + 1e-9)
                        elif self.args.target == 'logret':
                            log_ret = torch.log(ret_list + 1)
                            gt = 0
                            for i in range(step_idx, ret_list.shape[1]):
                                gt += log_ret[:, i] * (self.args.gamma ** (i - step_idx))
                            # gt = (gt - torch.mean(gt)) / (torch.std(gt) + 1e-9)
                        gt = gt.reshape(-1, 1)
                        loss = -torch.mean(gt * torch.log(torch.softmax(prob, dim=1)))
                        epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        print('loss', epoch_loss)
        epoch_ret = torch.cat(epoch_ret, dim=0)
        print('mean ret: ', torch.mean(epoch_ret))
        print('mean SR',
              (torch.mean(epoch_ret, dim=1) / torch.std(epoch_ret, dim=1)).mean() * np.sqrt(252 / self.args.max_steps))

    def get_episode_data(self, signal_matrix, ret_matrix, mask_matrix, batch_start_idx):
        # N x T x C
        # print(self.args.window_length, self.args.max_steps, batch_start_idx)
        batch_ret = ret_matrix[:, batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        # print(batch_ret.shape)
        batch_ret = batch_ret.to(self.device)
        batch_data = signal_matrix[:, batch_start_idx: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_mask = mask_matrix[:, batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_mask = batch_mask.to(self.device)
        ret_series = ret_matrix[:, batch_start_idx-1: batch_start_idx + self.args.window_length + self.args.max_steps - 2]
        # batch_date = date_matrix[:, batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        return batch_data, batch_ret, batch_mask, ret_series


    def train_episode_by_episode(self, signal_matrix, ret_matrix, mask_matrix, optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=False, return_weights=True, epoch=1):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[1]
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length + 2)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=1,
                                  stop=T - self.args.max_steps - self.args.window_length + 2,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
            # print(start_idx)
            # print(sample_num)
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_ret = []
        epoch_loss = []
        epoch_weight = []
        # print(signal_matrix.shape)
        # print(ret_matrix.shape)
        # print(mask_matrix.shape)
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_ret, batch_mask, batch_ret_series = self.get_episode_data(signal_matrix, ret_matrix, mask_matrix,
                                                                      batch_start_idx)
            # print(batch_data.shape, batch_ret.shape, batch_mask.shape, batch_ret_series.shape)

            batch_data = batch_data.to(self.device)
            batch_ret_series = batch_ret_series.unsqueeze(2)
            batch_ret_series = batch_ret_series.to(self.device)
            input_mask = None
            if self.args.model != 'mlp':
                input_mask = generate_lookback_mask(batch_data.shape[1], self.args.window_length, self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                N, T, C = batch_data.shape
                if self.args.model == 'mlp':
                    # output: T x N x 1
                    batch_data = batch_data.reshape(T * N, C).to(self.device)
                    scores = policy_net(batch_data, input_mask).reshape(T, N)
                else:
                    scores = policy_net(batch_data, batch_ret_series, input_mask)[:, -self.args.max_steps:].squeeze(2)
                    # print(scores.shape)
                if self.args.explore_noise and train:
                    noise = torch.randn(scores.shape).to(self.device)
                    scores = scores + noise * torch.std(scores) / epoch
                if self.args.weight_adj == 'raw':
                    # print(batch_mask.shape)
                    weight = scores * batch_mask
                    # print(weight.shape)
                elif self.args.weight_adj == 'adj':
                    weight = scores * batch_mask
                    weight = functions.torch_get_adjusted_weight(weight)
                elif self.args.weight_adj == 'self':
                    # scores = torch.sigmoid(scores)
                    scores = scores * batch_mask
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                # print(batch_ret.shape)
                cross_ret = torch.sum(batch_ret * weight, dim=0)
                # print(cross_ret.shape)
                epoch_weight.append(weight.detach().cpu().numpy())
                eq_weight_ret = torch.mean(batch_ret * batch_mask, dim=0)
                if self.args.target == 'sharpe':
                    gt = torch.mean(cross_ret) / torch.std(cross_ret)
                    eq_weight_sr = torch.mean(eq_weight_ret) / torch.std(eq_weight_ret)
                    if self.args.baseline:
                        gt -= eq_weight_sr
                elif self.args.target == 'ac_sharpe':
                    trading_cost = torch.sum(torch.cat([weight[:1], torch.abs(torch.diff(weight, dim=0))]) * 0.0015, dim=1)
                    ac_cross_ret = cross_ret - trading_cost
                    gt = torch.mean(ac_cross_ret) / torch.std(ac_cross_ret)
                if self.args.gradient_std == 'std':
                    loss = -torch.mean(gt) / self.args.batch_size
                else:
                    loss = -torch.mean(gt)
            if train:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.batch_size == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_ret.append(cross_ret.detach())
        epoch_loss = np.mean(epoch_loss)
        print('loss', epoch_loss)
        epoch_ret = torch.stack(epoch_ret)
        print(epoch_ret.shape)
        print('mean ret: ', torch.mean(epoch_ret))
        print('mean SR',
              (torch.mean(epoch_ret) / torch.std(epoch_ret)).mean() * np.sqrt(252 / self.args.max_steps))
        if return_weights:
            return epoch_loss, epoch_weight
        return epoch_loss
    



    def train(self):
        optimizer = self._select_optimizer()
        if self.args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,
                                                               verbose=True)
        elif self.args.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=(1e-7 / self.args.lr) ** (1 / self.args.epochs))
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(self.args.epochs):
            print('epoch', epoch + 1)
            print('======train=======')
            start_time = time.time()
            self.model.train()
            train_loss = self.train_episode_by_episode(self.train_data[0], self.train_data[1], self.train_data[2], optimizer, self.model, scaler,
                                                       sample_factor=self.args.sample_factor, train=True)
            self.model.eval()
            with torch.no_grad():
                print('======valid=======')
                valid_loss, valid_weight = self.train_episode_by_episode(self.valid_data[0], self.valid_data[1], self.valid_data[2], optimizer, self.model, scaler,
                                                           sample_factor=1, train=False, shuffle=False, epoch=epoch+1)
                print('======test=======')
                test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1], self.test_data[2], optimizer, self.model, scaler,
                                                          sample_factor=1, train=False, shuffle=False, epoch=epoch+1)
                if self.args.train_strategy == 'earlyStopping':
                    early_stopping(valid_loss, self.model, self.save_path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                if self.args.scheduler == 'plateau':
                    scheduler.step(valid_loss)
                elif self.args.scheduler == 'exponential':
                    scheduler.step()

            end_time = time.time()
            print('epoch time {0:.2f}'.format(end_time - start_time))
        if self.args.train_strategy == 'full':
            torch.save(self.model.state_dict(), self.save_path)
        self.model.load_state_dict(torch.load(self.save_path))
        # origin_policy = self.args.sample_policy
        # self.args.sample_policy = 'random'
        self.model.eval()
        test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1], self.test_data[2], optimizer,
                                                               self.model, scaler, train=False, shuffle=False, sample_factor=1, return_weights=True)
        # self.args.sample_policy = origin_policy
        # test_year = self.args.test_start_date.split('-')[0]
        # print(os.path.join(self.args.model_file_path, f'test_weight_{test_year}.pkl'))
        # pkl.dump(test_weight, open(os.path.join(self.args.model_file_path, f'test_weight_{test_year}.pkl'), 'wb'))
        return test_weight


    def train_episode_serially(self, signal_matrix, ret_matrix, mask_matrix, optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[1]
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length + 2)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=1,
                                  stop=T - self.args.max_steps - self.args.window_length + 2,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_ret = []
        epoch_loss = []
        epoch_weight = []
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_ret, batch_mask, batch_ret_series = self.get_episode_data(signal_matrix, ret_matrix, mask_matrix,
                                                                      batch_start_idx)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                N = batch_data.shape[0]
                weight_list = []
                for i in range(batch_data.shape[1]-self.args.window_length+1):
                    if not train:
                        with torch.no_grad():
                            slice_batch_data = batch_data[:, i:i+self.args.window_length]
                            ret_series = batch_ret_series[:, i:i+self.args.window_length]
                            slice_batch_data = slice_batch_data.to(self.device)
                            ret_series = ret_series.unsqueeze(2)
                            ret_series = ret_series.to(self.device)

                            # N x C
                            scores = policy_net(slice_batch_data, ret_series)
                            weight_list.append(scores)
                    else:
                        slice_batch_data = batch_data[:, i:i+self.args.window_length]
                        slice_batch_data = slice_batch_data.to(self.device)
                        ret_series = batch_ret_series[:, i:i+self.args.window_length]
                        ret_series = ret_series.unsqueeze(2)
                        ret_series = ret_series.to(self.device)
 
                        # N x C
                        scores = policy_net(slice_batch_data, ret_series)
                        weight_list.append(scores)
                    # output: N x 1
                scores = torch.cat(weight_list, dim=1)
                weight = scores * batch_mask
                cross_ret = torch.sum(batch_ret * weight, dim=0)
                epoch_weight.append(weight.detach().cpu().numpy())
                trading_cost = torch.sum(torch.cat([weight[:, :1], torch.abs(torch.diff(weight, dim=1))], dim=1) * 0.0015,
                                         dim=0)
                ac_cross_ret = cross_ret - trading_cost
                gt = torch.mean(ac_cross_ret) / torch.std(ac_cross_ret)
                if self.args.gradient_std == 'std':
                    loss = -torch.mean(gt) / self.args.batch_size
                else:
                    loss = -torch.mean(gt)
            if train:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_ret.append(cross_ret.detach())
        epoch_loss = np.mean(epoch_loss)
        print('loss', epoch_loss)
        epoch_ret = torch.stack(epoch_ret, dim=0)
        print('mean ret: ', torch.mean(epoch_ret))
        print('mean SR',
              (torch.mean(epoch_ret, dim=1) / torch.std(epoch_ret, dim=1)).mean() * np.sqrt(252 / self.args.max_steps))
        if return_weights:
            return epoch_loss, epoch_weight
        return epoch_loss


    def serial_train(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,
                                                                   verbose=True)
        elif self.args.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                               gamma=(1e-7 / self.args.lr) ** (1 / self.args.epochs))
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.epochs):
            print('epoch', epoch + 1)
            print('======train=======')
            start_time = time.time()
            self.model.train()
            train_loss = self.train_episode_serially(self.train_data[0], self.train_data[1], self.train_data[2],
                                                       optimizer, self.model, scaler,
                                                       sample_factor=self.args.sample_factor, train=True)
            self.model.eval()
            with torch.no_grad():
                print('======valid=======')
                valid_loss, valid_weight = self.train_episode_serially(self.valid_data[0], self.valid_data[1],
                                                                         self.valid_data[2], optimizer, self.model,
                                                                         scaler,
                                                                         sample_factor=1, train=False, shuffle=False,
                                                                         epoch=epoch + 1)
                print('======test=======')
                test_loss, test_weight = self.train_episode_serially(self.test_data[0], self.test_data[1],
                                                                       self.test_data[2], optimizer, self.model, scaler,
                                                                       sample_factor=1, train=False, shuffle=False,
                                                                       epoch=epoch + 1)
                if self.args.train_strategy == 'earlyStopping':
                    early_stopping(valid_loss, self.model, self.save_path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                if self.args.scheduler == 'plateau':
                    scheduler.step(valid_loss)
                elif self.args.scheduler == 'exponential':
                    scheduler.step()

            end_time = time.time()
            print('epoch time {0:.2f}'.format(end_time - start_time))
        if self.args.train_strategy == 'full':
            torch.save(self.model.state_dict(), self.save_path)
        self.model.load_state_dict(torch.load(self.save_path))
        origin_policy = self.args.sample_policy
        self.args.sample_policy = 'random'
        self.model.eval()
        test_loss, test_weight = self.train_episode_serially(self.test_data[0], self.test_data[1], self.test_data[2],
                                                               optimizer,
                                                               self.model, scaler, train=False, shuffle=False,
                                                               sample_factor=1, return_weights=True)
        self.args.sample_policy = origin_policy
        # test_year = self.args.test_start_date.split('-')[0]
        # print(os.path.join(self.args.model_file_path, f'test_weight_{test_year}.pkl'))
        # pkl.dump(test_weight, open(os.path.join(self.args.model_file_path, f'test_weight_{test_year}.pkl'), 'wb'))
        return test_weight

    def get_episode_data_with_gap(self, signal_matrix, ret_matrix, mask_matrix, batch_start_idx):
        # T x C x N or T x N x C
        batch_ret = ret_matrix[
                    batch_start_idx + self.args.window_length + self.args.pred_length - 2:
                    batch_start_idx + self.args.window_length + self.args.pred_length + self.args.max_steps - 2]
        batch_ret = batch_ret.to(self.device)
        batch_data = signal_matrix[batch_start_idx: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_data = batch_data.to(self.device)
        batch_mask = mask_matrix[
                     batch_start_idx + self.args.window_length + self.args.pred_length - 2:
                     batch_start_idx + self.args.window_length + self.args.pred_length + self.args.max_steps - 2]
        batch_mask = batch_mask.to(self.device)
        return batch_data, batch_ret, batch_mask

    def step1_train_with_gap(self, signal_matrix, ret_matrix, mask_matrix, optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[0]
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length - self.args.pred_length + 3)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=0,
                                  stop=T - self.args.max_steps - self.args.window_length - self.args.pred_length + 3,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_ret = []
        epoch_loss = []
        epoch_weight = []
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_ret, batch_mask = self.get_episode_data_with_gap(signal_matrix, ret_matrix, mask_matrix, batch_start_idx)
            input_mask = None
            if self.args.model != 'mlp':
                input_mask = generate_lookback_mask(batch_data.shape[0], self.args.window_length).to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                scores = policy_net(batch_data, input_mask)
                if self.args.mlp_implement == 'linear':
                    # output: T x N x 1
                    scores = scores.squeeze(dim=2)[self.args.window_length - 1:]
                if self.args.explore_noise and train:
                    noise = torch.randn(scores.shape).to(self.device)
                    scores = scores + noise * torch.std(scores) / epoch
                if self.args.weight_adj == 'raw':
                    weight = scores * batch_mask
                elif self.args.weight_adj == 'self':
                    scores = torch.sigmoid(scores)
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                cross_ret = torch.sum(batch_ret * weight, dim=1)
                epoch_weight.append(weight.detach().cpu().numpy())
                eq_weight_ret = torch.mean(batch_ret * batch_mask, dim=1)
                gt = torch.mean(cross_ret) / torch.std(cross_ret)
                eq_weight_sr = torch.mean(eq_weight_ret) / torch.std(eq_weight_ret)
                if self.args.baseline:
                    gt -= eq_weight_sr
                if self.args.gradient_std == 'std':
                    loss = -torch.mean(gt) / self.args.batch_size
                else:
                    loss = -torch.mean(gt)
            if train:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_ret.append(cross_ret.detach())
        epoch_loss = np.mean(epoch_loss)
        print('loss', epoch_loss)
        epoch_ret = torch.stack(epoch_ret, dim=0)
        print('mean ret: ', torch.mean(epoch_ret))
        print('mean SR',
              (torch.mean(epoch_ret, dim=1) / torch.std(epoch_ret, dim=1)).mean() * np.sqrt(252 / self.args.max_steps))
        if return_weights:
            return epoch_loss, epoch_weight
        return epoch_loss

    def step2_train_agg(self, signal_matrix, ret_matrix, mask_matrix, optimizer, pred_net_list, agg_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[0]
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length + 2)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=0,
                                  stop=T - self.args.max_steps - self.args.window_length + 2,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_ret = []
        epoch_loss = []
        epoch_weight = []
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_ret, batch_mask = self.get_episode_data(signal_matrix, ret_matrix, mask_matrix, batch_start_idx)
            input_mask = None
            if self.args.model != 'mlp':
                input_mask = generate_lookback_mask(batch_data.shape[0], self.args.window_length).to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                if self.args.freeze_param:
                    with torch.no_grad():
                        scores = pred_net_list(batch_data, input_mask)
                        # scores_list = []
                        # for pred_net in pred_net_list:
                        #     scores = pred_net(batch_data, input_mask)[self.args.window_length-1: ]
                        #     scores_list.append(scores)
                else:
                    scores = pred_net_list(batch_data, input_mask)
                #     for pred_net in pred_net_list:
                #         scores = pred_net(batch_data, input_mask)[self.args.window_length - 1:]
                #         scores_list.append(scores)
                # # output: T x N x K
                # scores = torch.cat(scores_list, dim=-1)
                # output: T x N x 1
                scores = agg_net(scores)
                scores = scores.squeeze(2)[self.args.window_length-1: ]
                if self.args.explore_noise and train:
                    noise = torch.randn(scores.shape).to(self.device)
                    scores = scores + noise * torch.std(scores) / epoch
                if self.args.weight_adj == 'raw':
                    weight = scores * batch_mask
                elif self.args.weight_adj == 'self':
                    scores = torch.sigmoid(scores)
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                cross_ret = torch.sum(batch_ret * weight, dim=1)
                trading_cost = torch.sum(torch.cat([weight[:1], torch.abs(torch.diff(weight, dim=0))]) * 0.0015, dim=1)
                ac_cross_ret = cross_ret - trading_cost
                epoch_weight.append(weight.detach().cpu().numpy())
                gt = torch.mean(ac_cross_ret) / torch.std(ac_cross_ret)
                if self.args.gradient_std == 'std':
                    loss = -torch.mean(gt) / self.args.batch_size
                else:
                    loss = -torch.mean(gt)
            if train:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_ret.append(cross_ret.detach())
        epoch_loss = np.mean(epoch_loss)
        print('loss', epoch_loss)
        epoch_ret = torch.stack(epoch_ret, dim=0)
        print('mean ret: ', torch.mean(epoch_ret))
        print('mean SR',
              (torch.mean(epoch_ret, dim=1) / torch.std(epoch_ret, dim=1)).mean() * np.sqrt(252 / self.args.max_steps))
        if return_weights:
            return epoch_loss, epoch_weight
        return epoch_loss

    def train_one_for_all(self):
        total_steps = self.args.pred_length
        pred_model_list = []
        for step in range(1, total_steps+1):
            self.args.pred_length = step
            self.model = MLP_linear(self.args, out_dim=1).to(self.device)
            self.save_path = f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_{step}.pt"
            base_path = '/mnt/HDD16TB/huahao/portfolio_construction/aim_port_model_file/twoStep_ac_sharpe_mlp_batch_size1_sample_factor0.1_max_steps20_predLen_10_roll_fixed_embedDim_128_2048_encLayer_4_drop_0.1_agg_{args.agg_num_layers}_{args.agg_embed_dim}'
            if os.path.exists(f"{base_path}/{self.args.test_start_date.split('-')[0]}_{step}.pt"):
                self.model.load_state_dict(torch.load(f"{base_path}/{self.args.test_start_date.split('-')[0]}_{step}.pt"))
                self.model.eval()
                pred_model_list.append(copy.deepcopy(self.model))
                continue
            optimizer = self._select_optimizer()
            if self.args.scheduler == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,
                                                                   verbose=True)
            elif self.args.scheduler == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                   gamma=(1e-7 / self.args.lr) ** (1 / self.args.epochs))
            scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            for epoch in range(self.args.epochs):
                print('epoch', epoch + 1)
                print('======train=======')
                start_time = time.time()
                self.model.train()
                train_loss = self.step1_train_with_gap(self.train_data[0], self.train_data[1], self.train_data[2], optimizer, self.model, scaler,
                                                           sample_factor=self.args.sample_factor, train=True)
                self.model.eval()
                with torch.no_grad():
                    print('======valid=======')
                    valid_loss, valid_weight = self.step1_train_with_gap(self.valid_data[0], self.valid_data[1], self.valid_data[2], optimizer, self.model, scaler,
                                                               sample_factor=1, train=False, shuffle=False, epoch=epoch+1)
                    print('======test=======')
                    test_loss, test_weight = self.step1_train_with_gap(self.test_data[0], self.test_data[1], self.test_data[2], optimizer, self.model, scaler,
                                                              sample_factor=1, train=False, shuffle=False, epoch=epoch+1)
                    if self.args.train_strategy == 'earlyStopping':
                        early_stopping(valid_loss, self.model, self.save_path)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break
                    if self.args.scheduler == 'plateau':
                        scheduler.step(valid_loss)
                    elif self.args.scheduler == 'exponential':
                        scheduler.step()

                end_time = time.time()
                print('epoch time {0:.2f}'.format(end_time - start_time))
            if self.args.train_strategy == 'full':
                torch.save(self.model.state_dict(), self.save_path)
            self.model.load_state_dict(torch.load(self.save_path))
            self.model.eval()
            # origin_policy = self.args.sample_policy
            self.args.sample_policy = 'random'
            pred_model_list.append(copy.deepcopy(self.model.cpu()))
        pred_model_list = Diff_gap_model(self.args, torch.nn.ModuleList(pred_model_list)).to(self.device)


        agg_save_path = f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_agg.pt"
        # 输入 T x N x K，输出T x N x 1
        agg_model = MLP_linear(args=None, out_dim=1, d_model=self.args.agg_embed_dim,
                               input_dim=self.args.pred_length, num_enc_layers=self.args.agg_num_layers, dropout=0.1, typical=False).to(self.device)

        if len(self.device_ids) > 1:
            pred_model_list = torch.nn.DataParallel(pred_model_list)
            agg_model = torch.nn.DataParallel(agg_model)
        if self.args.freeze_param:
            pred_model_list.eval()
            optimizer = torch.optim.Adam(agg_model.parameters(), lr=self.args.lr)
        else:
            pred_model_list.train()
            optimizer = torch.optim.Adam([
                {'params': pred_model_list.parameters(), 'lr': self.args.lr},
                {'params': agg_model.parameters(), 'lr': self.args.lr},
            ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(self.args.epochs):
            print('epoch', epoch + 1)
            print('======train=======')
            start_time = time.time()
            agg_model.train()
            pred_model_list.train()
            train_loss = self.step2_train_agg(self.train_data[0], self.train_data[1], self.train_data[2],
                                                    optimizer, pred_model_list, agg_model, scaler,
                                                    sample_factor=self.args.sample_factor, train=True)
            agg_model.eval()
            pred_model_list.eval()
            with torch.no_grad():
                print('======valid=======')
                valid_loss, valid_weight = self.step2_train_agg(self.valid_data[0], self.valid_data[1],
                                                                      self.valid_data[2], optimizer, pred_model_list, agg_model, scaler,
                                                                      sample_factor=1, train=False, shuffle=False,
                                                                      epoch=epoch + 1)
                print('======test=======')
                test_loss, test_weight = self.step2_train_agg(self.test_data[0], self.test_data[1],
                                                                    self.test_data[2], optimizer, pred_model_list, agg_model, scaler,
                                                                    sample_factor=1, train=False, shuffle=False,
                                                                    epoch=epoch + 1)
                if self.args.train_strategy == 'earlyStopping':
                    early_stopping(valid_loss, agg_model, agg_save_path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                scheduler.step(valid_loss)

            end_time = time.time()
            print('epoch time {0:.2f}'.format(end_time - start_time))
        with torch.no_grad():
            if self.args.train_strategy == 'full':
                torch.save(agg_model.state_dict(), agg_save_path)
            agg_model.load_state_dict(torch.load(agg_save_path))
            # origin_policy = self.args.sample_policy
            agg_model.eval()
            pred_model_list.eval()
            self.args.sample_policy = 'random'
            test_loss, test_weight = self.step2_train_agg(self.test_data[0], self.test_data[1], self.test_data[2],
                                                                optimizer,
                                                                pred_model_list, agg_model, scaler, train=False, shuffle=False,
                                                                sample_factor=1, return_weights=True)


        return test_weight


    def divide_train_valid_test(self, train_data, valid_data, test_data):
        test_start_idx = train_data.shape[1] + valid_data.shape[1]
        valid_start_idx = train_data.shape[1]
        new_data = np.concatenate([train_data, valid_data, test_data], axis=1)
        new_train = new_data[:, :valid_start_idx - self.args.window_length + 1]
        new_valid = new_data[:, valid_start_idx - self.args.window_length + 1: test_start_idx - self.args.window_length + 1]
        new_test = new_data[:, test_start_idx - self.args.window_length + 1:]
        return new_train, new_valid, new_test


    def _get_data_balanced(self):
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}_balanced.pkl', 'rb'))
        # N x T x C
        train_signal, train_ret, train_mask, train_time, train_stkcd = train
        valid_signal, valid_ret, valid_mask, valid_time, valid_stkcd = valid
        test_signal, test_ret, test_mask, test_time, test_stkcd = test
       
        train_signal, valid_signal, test_signal = self.divide_train_valid_test(train_signal, valid_signal, test_signal)
        train_ret, valid_ret, test_ret = self.divide_train_valid_test(train_ret, valid_ret, test_ret)
        train_mask, valid_mask, test_mask = self.divide_train_valid_test(train_mask, valid_mask, test_mask)
        train_time, valid_time, test_time = self.divide_train_valid_test(train_time, valid_time, test_time)
        train_stkcd, valid_stkcd, test_stkcd = self.divide_train_valid_test(train_stkcd, valid_stkcd, test_stkcd)
        train = (torch.from_numpy(train_signal), torch.from_numpy(train_ret), 
                 torch.from_numpy(train_mask), train_time, train_stkcd)
        valid = (torch.from_numpy(valid_signal), torch.from_numpy(valid_ret), 
                 torch.from_numpy(valid_mask), valid_time, valid_stkcd)
        test = (torch.from_numpy(test_signal), torch.from_numpy(test_ret), 
                torch.from_numpy(test_mask), test_time, test_stkcd)

        self.train_data = train
        self.valid_data = valid
        self.test_data = test

    def _merge_stkcd_mask_to_weight(self, weight_list):
        mask_matrix = self.test_data[2].detach().cpu().numpy()
        stkcd_matrix = self.test_data[4]
        date_matrix = self.test_data[3]
        T = mask_matrix.shape[1]
        start_idx = np.arange(start=1, stop=T - self.args.max_steps - self.args.window_length + 2,
                                    step=self.args.max_steps)
        sample_num = max(1, int(start_idx.shape[0]))
        date_list = []
        stkcd_list = []
        mask_list = []
        # print(date_matrix.shape, stkcd_matrix.shape, mask_matrix.shape) 
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_date = date_matrix[:, batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
            batch_stkcd = stkcd_matrix[:, batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
            batch_mask = mask_matrix[:, batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
            date_list.append(batch_date)
            stkcd_list.append(batch_stkcd)
            mask_list.append(batch_mask)
        date_matrix = np.concatenate(date_list, axis=1)
        stkcd_matrix = np.concatenate(stkcd_list, axis=1)
        mask_matrix = np.concatenate(mask_list, axis=1)
        weight = np.concatenate(weight_list, axis=1)
        df_list = []
        for day in range(date_matrix.shape[1]):
            df = pd.DataFrame({'date': date_matrix[:, day], 'stkcd': stkcd_matrix[:, day], 'weight': weight[:, day], 'mask': mask_matrix[:, day]})
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        # print(df.shape)
        # return df



   