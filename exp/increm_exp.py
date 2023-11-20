import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import BalancedDataset
from exp.base_exp import BaseExp
import pdb


'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class IncremExp(BaseExp):
    def __init__(self, args: dict[str, int], test_year: int, increase_date: str) -> None:
        super(IncremExp, self).__init__(args, test_year)
        self.args = args
        self.device = self._acquire_device()
        self.test_year = test_year
        self._get_data(test_year)
        self.increase_date = increase_date
        self.model = self._build_model().to(self.device)
        self.start_save_path = f"{self.args['model_file_path']}/{test_year}_checkpoint.pt"
        self.save_path = f"{self.args['model_file_path']}/{increase_date}_checkpoint.pt"
        self.test_data = test_data


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
        train, valid, test, basic_fea_num = pkl.load(open(f"{self.args['output_dataset_path']}/incremental_{self.cur_date}_{self.args['ret_type']}_{self.args['target_window']}d_{self.args['window_length']}.pkl", 'rb'))
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


    def train(self):
        self.model.load_state_dict(torch.load(self.start_save_path))
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
            print("Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss, model_optim.state_dict()['param_groups'][0]['lr']))
        early_stopping(0, self.model, self.save_path)
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        with torch.no_grad():
            df = self.process_one_epoch(self.test_loader, mode='test')
            df.to_pickle(f"{self.args['model_file_path']}/{self.cur_date}_test_result", )
        return self.model


