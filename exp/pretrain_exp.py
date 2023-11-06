import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN
from models.transformer.transformer import Transformer
from models.ffn import MLP_linear
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from utils.adjust_lr import adjust_learning_rate
from prepare_dataset.myDataset import RetDataset
from utils.loss_func import RetMSELoss
from utils.normalize import ts_mean_var_norm, ts_min_max_norm
import torch.multiprocessing as mp


'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class PretrainExp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        if len(self.device_ids) == 1:
            self._get_data()
            self.model = self._build_model().cuda()
        if self.args.est_method == 'rolling':
            self.save_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_PretrainCheckpoint.pt"
        else:
            self.save_path = f'{self.args.model_file_path}/PretrainCheckpoint.pt'
        self.finetune_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_FinetuneCheckpoint.pt"

    def save_args(self):
        if self.args.est_method == 'rolling':
            pkl.dump(self.args, open(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_config.pt", 'wb'))
        else:
            pkl.dump(self.args, open(f'{self.args.model_file_path}/config.pt', 'wb'))

    def _build_model(self, rank=0):
        if self.args.model == 'transformer':
            model = TransformerPretrain(self.args)
        if len(self.device_ids) > 1:
            model = model.to(rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        else:
            model = model.cuda()
        return model

    def _build_finetune_model(self, rank=None):
        if self.args.model == 'transformer':
            model = Transformer(self.args)
        state_dict = torch.load(self.save_path)
        model_dict = model.state_dict()
        state_dict = {item[0]: item[1] for item in state_dict.items() if 'backbone' in item[0]}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        if len(self.device_ids) > 1:
            model = model.to(rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        else:
            model = model.cuda()
        return model

    def _acquire_device(self):
        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]

        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def _get_data(self, rank=0):
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}.pkl', 'rb'))
        train_dataset = RetDataset(train, self.args)
        valid_dataset = RetDataset(valid, self.args)
        test_dataset = RetDataset(test, self.args)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def get_distributed_data(self, rank=0):
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'../data/{self.args.train_data_range}_{test_year}.pkl', 'rb'))
        train_dataset = RetDataset(train, self.args)
        valid_dataset = RetDataset(valid, self.args)
        test_dataset = RetDataset(test, self.args)
        dist.init_process_group(backend='nccl', world_size=len(self.device_ids), rank=rank, init_method='tcp://127.0.0.1:13456')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size//len(self.device_ids),
                                                        sampler=train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size//len(self.device_ids),
                                                        sampler=valid_sampler)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size//len(self.device_ids),
                                                       sampler=test_sampler)


    def _select_optimizer(self, params=None, lr=None):
        if self.args.optim == 'adam':
            return torch.optim.Adam(params if params else self.model.parameters(), lr=lr if lr else self.args.lr)
        elif self.args.optim == 'sgd':
            return torch.optim.SGD(params if params else self.model.parameters(), lr=lr if lr else self.args.lr)

    def random_masking(self, xb, mask_ratio):
        # xb: [N x L x D]
        N, L, D = xb.shape
        x = xb.clone()
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=xb.device)  # noise in [0, 1], N x L

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [N x L]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [N x len_keep x D]

        # removed x
        x_removed = torch.zeros(N, L - len_keep, D, device=xb.device)  # x_removed: [N x (L-len_keep) x D]
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [N x L x D]

        # combine the kept part and the removed one
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x nvars x patch_len]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)  # mask: [N x L]
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [N x L]
        # mask = mask.bool()
        return x_masked, x_kept, mask, ids_restore

    def process_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series) in enumerate(dataloader):
            signal = signal.reshape(-1, self.args.window_length, self.args.char_dim).cuda(non_blocking=True)
            ret_series = ret_series.reshape(-1, self.args.window_length, 1).cuda(non_blocking=True)
            signal_mask, _, rand_mask, _ = self.random_masking(signal, self.args.mask_ratio)
            mask = mask.reshape(-1, self.args.pred_length).squeeze(-1).cuda(non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                if self.args.mask_target == 'ret':
                    preds = self.model(signal_mask, ret_series)
                    loss = (preds - ret_series) ** 2
                    loss = loss.mean(dim=-1)
                    loss = (loss * rand_mask).sum(dim=-1) / rand_mask.sum(dim=-1)
                    loss = (loss * mask).sum() / mask.sum()
                elif self.args.mask_target == 'signal':
                    preds = self.model(signal_mask, ret_series)
                    # N x L x D
                    loss = (preds - signal) ** 2
                    loss = loss.mean(dim=-1)
                    loss = (loss * rand_mask).sum(dim=-1) / rand_mask.sum(dim=-1)
                    loss = (loss * mask).sum() / mask.sum()
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

    def train(self, proc=0, nprocs=0, arg=None):
        if len(self.device_ids) > 1:
            torch.cuda.set_device(proc)
            self.get_distributed_data(rank=proc)
            self.model = self._build_model(proc)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, distributed=True)
        else:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, distributed=False)
        train_steps = len(self.train_loader)
        self.args.match_stkcd_date = False
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
            train_loss = self.process_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            self.model.eval()
            with torch.no_grad():
                vali_loss = self.process_one_epoch(self.valid_loader, criterion, train=False)
            scheduler.step(vali_loss)
            print("Pretrain Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f} | Iter count: {5}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss, model_optim.state_dict()['param_groups'][0]['lr'], iter_count))
            if proc == 0:
                # torch.save(self.model.module.state_dict(), self.save_path)
                early_stopping(vali_loss, self.model, self.save_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    return

    def finetune_one_epoch(self, dataloader, criterion, scaler=None, optimizer=None, train=False):
        total_loss = []
        for i, (signal, ret, mask, ret_series) in enumerate(dataloader):
            signal = signal.reshape(-1, self.args.window_length, self.args.char_dim).to(self.device)
            mask = mask.reshape(-1, self.args.pred_length).squeeze(-1).to(self.device)
            ret_series = ret_series.reshape(-1, self.args.window_length, 1).to(self.device)
            ret = ret.reshape(-1, self.args.pred_length).to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                preds = self.model(signal, ret_series)
                loss = (preds - ret) ** 2
                loss = loss.mean(dim=-1)
                loss = (loss * mask).sum() / mask.sum()
                if train:
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def finetune(self, proc=0, nprocs=0, arg=None):
        if len(self.device_ids) > 1:
            torch.cuda.set_device(proc)
            self.get_distributed_data(rank=proc)
        self.model = self._build_finetune_model(proc)

        train_steps = len(self.train_loader)
        self.args.match_stkcd_date = False
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(None, self.args.lr * 2e-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.2, patience=1,
                                                               verbose=True)
        criterion = torch.nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(10):
            iter_count = 0
            epoch_time = time.time()
            '''
            signal: (batch_size, num_stock, seq_len, feature_dim)
            ret_series: (batch_size, num_stock, seq_len, 1)
            mask: (batch_size, num_stock, seq_len)
            '''
            iter_count += 1
            self.model.train()
            train_loss = self.finetune_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            self.model.eval()
            with torch.no_grad():
                vali_loss = self.finetune_one_epoch(self.valid_loader, criterion, train=False)
            scheduler.step(vali_loss)
            print("Fintune Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss,
                model_optim.state_dict()['param_groups'][0]['lr']))
            if proc == 0:
                if len(self.device_ids) > 1:
                    torch.save(self.model.module.state_dict(), self.finetune_path)
                else:
                    torch.save(self.model.state_dict(), self.finetune_path)
                # early_stopping(vali_loss, self.model, self.finetune_path)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     break
        # self.model.load_state_dict(torch.load(self.finetune_path))
        # return self.model


    def linear_probing(self):
        train_steps = len(self.train_loader)
        self.model = self._build_finetune_model().to(self.device)
        self.args.match_stkcd_date = False
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(self.model.head.parameters(), self.args.lr * 1e-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.2, patience=1,
                                                               verbose=True)
        criterion = torch.nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(10):
            iter_count = 0
            epoch_time = time.time()
            '''
            signal: (batch_size, num_stock, seq_len, feature_dim)
            ret_series: (batch_size, num_stock, seq_len, 1)
            mask: (batch_size, num_stock, seq_len)
            '''
            iter_count += 1
            self.model.train()
            train_loss = self.finetune_one_epoch(self.train_loader, criterion, scaler, model_optim, train=True)
            self.model.eval()
            with torch.no_grad():
                vali_loss = self.finetune_one_epoch(self.valid_loader, criterion, train=False)
            scheduler.step(vali_loss)
            print("Fintune Epoch: {0}, Cost Time: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | LR: {4:.7f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_loss,
                model_optim.state_dict()['param_groups'][0]['lr']))
            torch.save(self.model.state_dict(), self.finetune_path)
            # early_stopping(vali_loss, self.model, self.finetune_path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
        self.model.load_state_dict(torch.load(self.finetune_path))
        return self.model

    def test_model(self, proc=0, nprocs=0, arg=None):
        self.args.match_stkcd_date = True
        if self.args.model == 'transformer':
            model = Transformer(self.args)
        state_dict = torch.load(self.finetune_path)
        model.load_state_dict(state_dict)
        if len(self.device_ids) > 1:
            torch.cuda.set_device(proc)
            self.get_distributed_data(rank=proc)
            model = model.to(proc)
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[proc])
        else:
            self.model = model.to(self.device)
        self.model.eval()
        df_list = []
        with torch.no_grad():
            for i, (data) in enumerate(self.test_loader):
                signal = data[0].reshape(-1, self.args.window_length, self.args.char_dim)
                mask = data[2].reshape(-1, self.args.pred_length)
                signal = signal.to(self.device)
                ret_series = data[3].reshape(-1, self.args.window_length, 1).to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
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
        print(df.shape)
        if self.args.est_method == 'rolling':
            df.to_pickle(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_pred_result", )
        else:
            df.to_pickle(f'{self.args.model_file_path}/pred_result', )

