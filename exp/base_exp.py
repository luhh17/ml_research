import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN, LSTM
from models.transformer.transformer import Transformer
from models.targetformer.targetformer import Targetformer
from models.ffn import Mlp3D, Mlp2D, MLP_linear_concat
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import BalancedDataset
from utils.loss_func import sharpe_loss, mseic_loss
import pdb


'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class BaseExp(object):
    def __init__(self, args: dict[str, int], test_year: int, test_data: list=None) -> None:
        self.args = args
        self.device = self._acquire_device()
        self.test_year = test_year
        if not test_data:
            self._get_data(test_year)
        self.model = self._build_model().to(self.device)
        self.save_path = f"{self.args['model_file_path']}/{test_year}_checkpoint.pt"
        self.test_data = test_data

    def save_args(self) -> None:
        pkl.dump(self.args, open(f"{self.args['model_file_path']}/config", 'wb'))

    def _build_model(self) -> nn.Module:
        if self.args['model'] == 'rnn':
            model = RNN(self.args)
        elif self.args['model'] == 'lstm':
            model = LSTM(self.args)
        elif self.args['model'] == 'transformer':
            model = Transformer(self.args)
        elif self.args['model'] == 'targetformer':
            model = Targetformer(self.args)
        elif self.args['model'] == 'mlp':
            # if self.args.mode=='dualnet':
            #     model = MLP_linear_concat(self.args, out_dim=self.args.pred_length)
            # else:
            model = Mlp3D(self.args, out_dim=self.args['pred_length'])
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _acquire_device(self) -> torch.device:
        device_ids = self.args['devices'].split(',')
        self.device_ids = [int(id_) for id_ in device_ids]
        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args['devices']))
        return device

    def _get_data(self, test_year: int) -> None:
        """
        load data and return dataloader
        Parameters:
            test_year: int
        Returns:
            train_loader: torch.utils.data.DataLoader
            valid_loader: torch.utils.data.DataLoader
            test_loader: torch.utils.data.DataLoader
        """
        print(f'loading data {test_year}')
        train, valid, test, basic_fea_num = pkl.load(open(f"{self.args['output_dataset_path']}/roll_{test_year}_{self.args['ret_type']}_{self.args['target_window']}d_{self.args['window_length']}_balanced.pkl", 'rb'))
        self.args['char_dim'] = basic_fea_num
        train_dataset = BalancedDataset(train, self.args, match_stkcd_date=False)
        valid_dataset = BalancedDataset(valid, self.args, match_stkcd_date=False)    
        test_dataset = BalancedDataset(test, self.args, match_stkcd_date=True)
        del train,valid,test
        if self.args['loss_function']=='Mse' or self.args['loss_function']=='MseIc':
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True)
        elif self.args['loss_function']=='Sharpe':
            # don't shuffle the samples to ensure the economic meaning of sharpe ratio
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args['batch_size'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False)

        del train_dataset, valid_dataset, test_dataset
        print('loading data done!')
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        return


    def _select_optimizer(self) -> torch.optim.Optimizer:
        if self.args['optim'] == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        elif self.args['optim']== 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args['lr'])


    def form_batch_test_res(self, preds, data, mask):
        """
        Reformat and save the test result
        Parameters:
            preds: torch.tensor
            data: tuple
            mask: torch.tensor
        Returns:
            ret: np.array
            stkcd: np.array
            time: np.array
            mask: np.array
            df: pd.DataFrame
        """
        preds = preds.detach().cpu().numpy().reshape(-1)
        ret = data[1].detach().cpu().numpy().reshape(-1)
        mask = mask.detach().cpu().numpy()
        mask = data[2].detach().cpu().numpy().reshape(-1)
        
        time = data[4].detach().cpu().numpy().reshape(-1)
        stkcd = data[5].detach().cpu().numpy().reshape(-1)

        ret = ret[mask == 1]
        stkcd = stkcd[mask == 1]
        time = time[mask == 1]
        preds = preds[mask == 1]
        mask = mask[mask == 1]
        df_dict = {'stkcd': stkcd, 'date': time}
        df_dict[f'ret'] = ret.squeeze()
        df_dict[f'pred'] = preds.squeeze()
        df_dict[f'mask'] = mask.squeeze()
        df = pd.DataFrame(df_dict)
        return ret, stkcd, time, mask, df


    def form_one_batch_data(self, data):
        """
        Reformat the data
        Parameters:
            data: tuple
        Returns:
            signal: torch.tensor
            ret: torch.tensor
            mask: torch.tensor
            ret_series: torch.tensor
        """
        signal = data[0].reshape(-1, self.args['window_length'], self.args['char_dim']).to(self.device)
        ret = data[1].reshape(-1, self.args['pred_length']).to(self.device)
        mask = data[2].reshape(-1, self.args['pred_length']).to(self.device)
        ret_series = data[3].reshape(-1, self.args['window_length'], 1).to(self.device)
        return signal, ret, mask, ret_series


    def process_one_epoch(self, dataloader, scaler=None, optimizer=None, mode: str='train'):
        """
        train, valid, test all use this function for iterate one epoch
        Parameters: 
            dataloader: torch.utils.data.DataLoader
            scaler: torch.cuda.amp.GradScaler
            optimizer: torch.optim.Optimizer
            mode: str
        Returns:
            total_loss: float
        """
        total_loss = []
        df_list = []
        criterion = torch.nn.MSELoss()
        for i, data in enumerate(dataloader):
            signal, ret, mask, ret_series = self.form_one_batch_data(data)
            batch_size = signal.shape[0]
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args['use_amp']):
                preds = self.model(signal, y_enc=ret_series)
                preds = preds.reshape(-1, self.args['pred_length'])
                preds = preds * mask
                ret = ret * mask
                if self.args['loss_function'] == 'Sharpe':
                    loss = sharpe_loss(preds, ret, batch_size)
                elif self.args['loss_function'] == 'MseIc':
                    loss = mseic_loss(preds, ret, mask, batch_size)
                elif self.args['loss_function'] == 'Mse':
                    loss = criterion(preds, ret)

            if mode == 'train':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss.append(loss.item())
            if mode == 'test':
                ret, stkcd, time, mask, df = self.form_batch_test_res(preds, data, mask)
                df_list.append(df)
        total_loss = np.average(total_loss)
        if mode == 'test':
            df = pd.concat(df_list)
            return df
        return total_loss


    def train(self):
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args['patience'], verbose=True)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.2, patience=4, verbose=True)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args['epochs']):
            iter_count = 0
            epoch_time = time.time()
            '''
            signal: (batch_size, num_stock, seq_len, feature_dim)
            ret_series: (batch_size, num_stock, seq_len, 1)
            mask: (batch_size, num_stock, seq_len)
            '''
            iter_count += 1
            self.model.train()
            train_loss = self.process_one_epoch(self.train_loader, scaler, model_optim, mode='train')
            self.model.eval()
            with torch.no_grad():
                vali_loss = self.process_one_epoch(self.valid_loader, mode='valid')
            scheduler.step(vali_loss)
            print("Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss, model_optim.state_dict()['param_groups'][0]['lr']))
            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        with torch.no_grad():
            df = self.process_one_epoch(self.test_loader, mode='test')
            df.to_pickle(f"{self.args['model_file_path']}/{self.test_year}_pred_result", )
        return self.model

