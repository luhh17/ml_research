import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN
from models.transformer import Transformer, Informer
from models.ffn import MLP_linear, MLP_ts, MLP_post_ts, MLP_RNN
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import RetDataset, divide_train_valid_test, FactorDataset
from utils.loss_func import RetMSELoss
from utils.normalize import ts_mean_var_norm, ts_min_max_norm
'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class FactorExp(object):
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
        elif self.args.model == 'mlp':
            if self.args.contain_alpha:
                model = MLP_linear(self.args, out_dim=4)
            else:
                model = MLP_linear(self.args, out_dim=3)
        elif self.args.model == 'mlp_ts':
            model = MLP_ts(self.args, out_dim=self.args.pred_length)
        elif self.args.model == 'mlp_post_ts':
            model = MLP_post_ts(self.args, out_dim=self.args.pred_length)
        elif self.args.model == 'mlp_rnn':
            model = MLP_RNN(self.args, out_dim=self.args.pred_length)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _acquire_device(self):
        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]

        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def _get_data(self):
        test_year = self.args.test_start_date.split('-')[0]
        if self.args.infer_factor:
            if self.args.nolead:
                train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}_nolead.pkl', 'rb'))
            else:
                train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}.pkl', 'rb'))
            train_dataset = RetDataset(train, self.args)
            valid_dataset = RetDataset(valid, self.args)
            test_dataset = RetDataset(test, self.args)
        else:
            if self.args.nolead:
                train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}_ff3_nolead', 'rb'))
            else:
                train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}_ff3', 'rb'))
            train_dataset = FactorDataset(train, self.args)
            valid_dataset = FactorDataset(valid, self.args)
            test_dataset = FactorDataset(test, self.args)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

    def infer_factor_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series) in enumerate(dataloader):
            signal = signal.reshape(-1, self.args.window_length, self.args.char_dim).to(self.device)
            mask = mask.to(self.device)
            ret_series = ret_series.reshape(-1, self.args.window_length, 1).to(self.device)
            ret = ret.to(self.device)
            pred_beta = self.model(signal, ret_series)
            batch_num = mask.shape[0]
            if self.args.contain_alpha:
                preds = preds
            else:
                x = pred_beta.reshape(batch_num, -1, self.args.factor_num)
                b = torch.linalg.inv(x.transpose(1, 2) @ x) @ (x.transpose(1, 2) @ ret)
                resid = ret - x @ b
                preds = x @ b
                mask_ivol = mask.float() @ mask.float().transpose(1, 2)
                ivol = (resid @ resid.transpose(1, 2)) * mask_ivol
                ivol[:, np.arange(ivol.shape[1]), np.arange(ivol.shape[1])] = np.nan
                mean_ivol = torch.nanmean(torch.abs(ivol), dim=(1, 2))
                ivol_loss = torch.nanmean(mean_ivol)
                preds = preds * mask
                ret = ret * mask
                mse_loss = criterion(preds, ret)

                loss = mse_loss + ivol_loss * self.args.ivol_factor
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                optimizer.step()
                optimizer.zero_grad()
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def process_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series, factors) in enumerate(dataloader):
            signal = signal.reshape(-1, self.args.window_length, self.args.char_dim)
            factors = factors.reshape(-1, 3).to(self.device)
            mask = mask.reshape(-1, self.args.pred_length)
            signal = signal.to(self.device)
            ret_series = ret_series.reshape(-1, self.args.window_length, 1).to(self.device)
            ret = ret.reshape(-1, self.args.pred_length).to(self.device)
            mask = mask.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                pred_beta = self.model(signal, ret_series)
                if self.args.contain_alpha:
                    preds = torch.sum(pred_beta[:, :3] * factors, dim=1, keepdim=True)
                    alpha = pred_beta[:, 3:]
                    preds = preds + alpha
                else:
                    preds = pred_beta * factors
                    preds = torch.sum(preds, dim=1, keepdim=True)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.2, patience=2, verbose=True)
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
            if self.args.infer_factor:
                train_loss = self.infer_factor_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
                self.model.eval()
                with torch.no_grad():
                    vali_loss = self.infer_factor_one_epoch(self.valid_loader, criterion, train=False)
            else:
                train_loss = self.process_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
                self.model.eval()
                with torch.no_grad():
                    vali_loss = self.process_one_epoch(self.valid_loader, criterion, train=False)
                # test_loss = self.process_one_epoch(self.test_loader, criterion, train=False)
            scheduler.step(vali_loss)
            print("Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss, model_optim.state_dict()['param_groups'][0]['lr']))
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
                signal = data[0].reshape(-1, self.args.window_length, self.args.char_dim).to(self.device)
                mask = data[2].reshape(-1, self.args.pred_length).to(self.device)
                factors = data[6].reshape(-1, 3).to(self.device)
                ret_series = data[3].reshape(-1, self.args.window_length, 1).to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                    pred_beta = self.model(signal, ret_series)
                    if self.args.contain_alpha:
                        preds = torch.sum(pred_beta[:, :3] * factors, dim=1, keepdim=True)
                        alpha = pred_beta[:, 3:]
                        preds = preds + alpha
                    else:
                        preds = pred_beta * factors
                        preds = torch.sum(preds, dim=1, keepdim=True)
                    preds = preds * mask
                preds = preds.detach().cpu().numpy().reshape(-1, self.args.pred_length)
                ret = data[1].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                mask = data[2].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                if self.args.contain_alpha:
                    pred_beta = pred_beta.detach().cpu().numpy().reshape(-1, 4)
                else:
                    pred_beta = pred_beta.detach().cpu().numpy().reshape(-1, 3)
                factor = data[6].detach().cpu().numpy().reshape(-1, 3)
                if self.args.match_stkcd_date:
                    time = data[4].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                    stkcd = data[5].detach().cpu().numpy().reshape(-1)
                    df_dict = {'stkcd': stkcd, 'date': time[:, 0]}
                    df_dict[f'ret'] = ret.squeeze()
                    df_dict[f'mask'] = mask.squeeze()
                    df_dict['pred'] = preds.squeeze()
                    if self.args.contain_alpha:
                        df_dict['pred_alpha'] = pred_beta[:, 3]
                    for i in range(3):
                        df_dict[f'pred_beta{i}'] = pred_beta[:, i]
                        df_dict[f'factor{i}'] = factor[:, i]
                    df = pd.DataFrame(df_dict)
                    df_list.append(df)
        df = pd.concat(df_list)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result',)

    def infer_factor_test_model(self):
        self.args.match_stkcd_date = True
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        df_list = []
        with torch.no_grad():
            for i, (data) in enumerate(self.test_loader):
                signal = data[0].reshape(-1, self.args.window_length, self.args.char_dim).to(self.device)
                mask = data[2].to(self.device)
                ret = data[1].to(self.device)
                ret_series = data[3].reshape(-1, self.args.window_length, 1).to(self.device)
                pred_beta = self.model(signal, ret_series)
                batch_num = mask.shape[0]
                if self.args.contain_alpha:
                    continue
                else:
                    x = pred_beta.reshape(batch_num, -1, self.args.factor_num)
                    b = torch.linalg.inv(x.transpose(1, 2) @ x) @ (x.transpose(1, 2) @ ret)
                    preds = x @ b
                preds = preds * mask
                preds = preds.detach().cpu().numpy().reshape(-1, self.args.pred_length)
                ret = data[1].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                mask = data[2].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                if self.args.contain_alpha:
                    pred_beta = pred_beta.detach().cpu().numpy().reshape(-1, 4)
                else:
                    pred_beta = pred_beta.detach().cpu().numpy().reshape(-1, 3)
                if self.args.match_stkcd_date:
                    time = data[4].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                    stkcd = data[5].detach().cpu().numpy().reshape(-1)
                    df_dict = {'stkcd': stkcd, 'date': time[:, 0]}
                    df_dict[f'ret'] = ret.squeeze()
                    df_dict[f'mask'] = mask.squeeze()
                    df_dict['pred'] = preds.squeeze()
                    if self.args.contain_alpha:
                        df_dict['pred_alpha'] = pred_beta[:, 3]
                    for i in range(self.args.factor_num):
                        df_dict[f'pred_beta{i}'] = pred_beta[:, i]
                    df = pd.DataFrame(df_dict)
                    df_list.append(df)
        df = pd.concat(df_list)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result',)



