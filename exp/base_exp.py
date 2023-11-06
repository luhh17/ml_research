import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN,LSTM
from models.transformer.transformer import Transformer
from models.targetformer.targetformer import Targetformer
from models.autoformer.autoformer_backbone import AutoFormer
from models.ffn import MLP_linear, MLP_ts, MLP_post_ts, MLP_RNN
from models.ffn import MLP_linear_concat
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import RetDataset, BalancedDataset
from utils.loss_func import RetMSELoss
from utils.normalize import ts_mean_var_norm, ts_min_max_norm
from lossfunc import Sharpe_loss,MSEIC_loss
import pdb

'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class Base_Exp(object):
    def __init__(self, args,exp_loader):
        self.args = args
        self.device = self._acquire_device()
        self._get_data(exp_loader)
        self.model = self._build_model().to(self.device)
        # dist.init_process_group(backend='nccl', init_method='env://')
        self.save_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_checkpoint.pt"
        self.model = self.model.to(self.device)

    def save_args(self):
        if self.args.est_method == 'rolling':
            pkl.dump(self.args, open(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_config.pt", 'wb'))
        else:
            pkl.dump(self.args, open(f'{self.args.model_file_path}/config.pt', 'wb'))

    def _build_model(self):
        if self.args.model == 'rnn':
            model = RNN(self.args)
        elif self.args.model == 'lstm':
            model = LSTM(self.args)
        elif self.args.model == 'transformer':
            model = Transformer(self.args)
        elif self.args.model == 'targetformer':
            model = Targetformer(self.args)
        # elif self.args.model == 'informer':
        #     model = Informer(self.args)
        elif self.args.model == 'autoformer':
            model = AutoFormer(self.args)
        elif self.args.model == 'mlp':
            if self.args.mode=='dualnet':
                model=MLP_linear_concat(self.args, out_dim=self.args.pred_length)
            else:
                model = MLP_linear(self.args, out_dim=self.args.pred_length)
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

    def divide_train_valid_test(self, train_data, valid_data, test_data):
        test_start_idx = train_data.shape[1] + valid_data.shape[1]
        valid_start_idx = train_data.shape[1]
        new_data = np.concatenate([train_data, valid_data, test_data], axis=1)
        if self.args.subsample:
            new_train = new_data[:200, :valid_start_idx - self.args.window_length + 1]
            new_valid = new_data[:200,
                        valid_start_idx - self.args.window_length + 1: test_start_idx - self.args.window_length + 1]
            new_test = new_data[:200, test_start_idx - self.args.window_length + 1:]
        else:
            new_train = new_data[:, :valid_start_idx - self.args.window_length + 1]
            new_valid = new_data[:, valid_start_idx - self.args.window_length + 1: test_start_idx - self.args.window_length + 1]
            new_test = new_data[:, test_start_idx - self.args.window_length + 1:]
        return new_train, new_valid, new_test

    def _get_data(self,exp_loader):
        self.train_loader ,self.valid_loader,self.test_loader = exp_loader

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

    def process_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series) in enumerate(dataloader):
            batch_size=signal.shape[0]
            signal = signal.reshape(-1, self.args.window_length, self.args.char_dim)
            mask = mask.reshape(-1, self.args.pred_length)
            signal = signal.to(self.device)
            ret_series = ret_series.reshape(-1, self.args.window_length, 1).to(self.device)
            ret = ret.reshape(-1, self.args.pred_length).to(self.device)
            mask = mask.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                if self.args.model == 'autoformer':
                    dec_inp = torch.zeros_like(ret_series[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([ret_series[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    preds = self.model(signal, None, dec_inp, None)
                else:
                    preds = self.model(signal, y_enc=ret_series)
                preds = preds.reshape(-1, self.args.pred_length)
                preds = preds * mask
                ret = ret * mask

                if self.args.loss_function=='Sharpe':
                    loss = Sharpe_loss(preds,ret,batch_size)
                    
                elif self.args.loss_function=='MSEIC':
                    loss = MSEIC_loss(preds,ret,mask,batch_size)
                    
                else:
                    loss = criterion(preds, ret)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss.append(loss.item())
        total_loss = np.nanmean(total_loss)
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
            train_loss = self.process_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            self.model.eval()
            with torch.no_grad():
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
        print(self.args.pred_length)
        with torch.no_grad():
            for i, (data) in enumerate(self.test_loader):
                signal = data[0].reshape(-1, self.args.window_length, self.args.char_dim)
                mask = data[2].reshape(-1, self.args.pred_length)
                signal = signal.to(self.device)
                ret_series = data[3].reshape(-1, self.args.window_length, 1).to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                    if self.args.model == 'autoformer':
                        dec_inp = torch.zeros_like(ret_series[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([ret_series[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        preds = self.model(signal, None, dec_inp, None)
                    else:
                        preds = self.model(signal, ret_series)
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



