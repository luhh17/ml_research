import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.tempflow.tempflow_network import TempFlowNetwork
from models.ffn import MLP, MLP_linear
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import RetDataset, divide_train_valid_test, select_subsample, ConstantMaskDataset
from utils.loss_func import RetMSELoss
'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class MultiExp(object):
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
        if self.args.model == 'tempflow':
            model = TempFlowNetwork(self.args)
        elif self.args.model == 'mlp':
            model = MLP_linear(args=None, out_dim=1, d_model=256, input_dim=self.args.window_length-1, num_enc_layers=3)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _acquire_device(self):
        self.args.devices = self.args.devices.replace(' ', '')
        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def _get_data(self):
        test_year = self.args.test_start_date.split('-')[0]
        select_subsample(self.args)
        # train, valid, test = pkl.load(open(f'../data/hs300_{test_year}.pkl', 'rb'))
        # train_dataset = RetDataset(train, self.args)
        # valid_dataset = RetDataset(valid, self.args)
        # test_dataset = RetDataset(test, self.args)
        # self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True)
        # self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True)
        # self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True)

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

    def process_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        mse_total = []
        for i, (signal, ret, mask, ret_series) in enumerate(dataloader):
            # B x N x T x C
            # B x N x (T-1)
            signal = signal.to(self.device)
            ret_series = ret_series.to(self.device)
            ret = ret.to(self.device)
            mask = mask.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                if self.args.model == 'tempflow':
                    loss, mse_loss, pred_list = self.model(past_time_series=ret_series, target=ret, past_covariates=signal, mask=mask)
                elif self.args.model == 'mlp':
                    ret_series = ret_series.reshape(-1, self.args.window_length-1)
                    ret = ret.reshape(-1, self.args.pred_length)
                    mask = mask.reshape(-1, self.args.pred_length)
                    preds = self.model(ret_series)
                    preds = preds * mask
                    loss = criterion(preds, ret)
                    mse_loss = loss
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            total_loss.append(loss.item())
            mse_total.append(mse_loss.item())
        total_loss = np.average(total_loss)
        mse_total = np.average(mse_total)
        return total_loss, mse_total

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
            train_loss, train_mse = self.process_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            self.model.eval()
            with torch.no_grad():
                vali_loss, valid_mse = self.process_one_epoch(self.valid_loader, criterion, train=False)
                # test_loss = self.process_one_epoch(self.test_loader, criterion, train=False)
            scheduler.step(valid_mse)
            print("Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Train MSE: {3:.7f} Vali Loss: {4:.7f} Vali MSE: {5:.7f} | LR: {6:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, train_mse, vali_loss, valid_mse, model_optim.state_dict()['param_groups'][0]['lr']))
            early_stopping(valid_mse, self.model, self.save_path)
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
                signal = data[0].to(self.device)
                ret_series = data[3].to(self.device)
                mask = data[2].to(self.device)
                ret = data[1].to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                    if self.args.model == 'tempflow':
                        loss, mse_loss, pred_list = self.model(past_time_series=ret_series, target=ret, past_covariates=signal, mask=mask)
                    elif self.args.model == 'mlp':
                        ret_series = ret_series.reshape(-1, self.args.window_length - 1)
                        mask = mask.reshape(-1, self.args.pred_length)
                        preds = self.model(ret_series)
                        preds = preds * mask
                        pred_list = [preds]
                pred_list = [pred.detach().cpu().numpy().reshape(-1, self.args.pred_length) for pred in pred_list]
                ret = data[1].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                mask = data[2].detach().cpu().numpy().reshape(-1, self.args.pred_length)

                if self.args.match_stkcd_date:
                    time = data[4].detach().cpu().numpy().reshape(-1, self.args.pred_length)
                    stkcd = data[5].detach().cpu().numpy().reshape(-1)
                    df_dict = {'stkcd': stkcd, 'date': time[:, 0]}
                    if self.args.model == 'mlp':
                        df_dict['pred'] = pred_list[0][:, 0]
                    else:
                        for i in range(100):
                            df_dict[f'pred_{i}'] = pred_list[i][:, 0]
                    df_dict[f'ret'] = ret[:, 0]
                    df_dict[f'mask'] = mask[:, 0]
                    df = pd.DataFrame(df_dict)
                    df_list.append(df)
        df = pd.concat(df_list)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result',)



