import numpy as np
import torch
from exp.rl_exp import RL_Exp
import time
from utils.early_stop import EarlyStopping
import torch_package.functions as functions
import pickle as pkl
from models.ffn import MLP_conv, MLP_linear, serial_MLP, Diff_gap_model
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F


class CustomImageDataset(Dataset):
    def __init__(self, args, signal, label, leak_length=1):
        T, N, C = signal.shape
        stop = np.arange(start=0,
                                  stop=T - args.max_steps - args.window_length + 2 - leak_length,
                                  step=args.max_steps)[-1] + args.max_steps
        signal = signal[:stop]
        T, N, C = signal.shape
        print(signal.dtype)
        print(label.dtype)
        self.signal = signal.reshape(T * N, C)
        self.label = label.reshape(T * N)
        self.sample_num = T * N

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        signal = self.signal[idx]
        label = self.label[idx]
        return signal, label


class LeakageTrain(RL_Exp):
    def __init__(self, args):
        super().__init__(args)
        self.target_data_path = '/mnt/HDD16TB/huahao/portfolio_construction/leak/mlp_2D'


    def get_episode_data(self, signal_matrix, ret_matrix, mask_matrix, batch_start_idx, leak_length=1):
        # T x C x N or T x N x C
        batch_ret = ret_matrix[
                    batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_ret = batch_ret.to(self.device)
        batch_data = signal_matrix[batch_start_idx + leak_length: batch_start_idx + self.args.window_length + self.args.max_steps - 1 + leak_length]
        batch_data = batch_data.to(self.device)
        batch_mask = mask_matrix[
                     batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_mask = batch_mask.to(self.device)
        return batch_data, batch_ret, batch_mask

    def train_episode_by_episode(self, signal_matrix, ret_matrix, mask_matrix, optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1, leak_length=1):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[0]
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length + 2 - leak_length)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=0,
                                  stop=T - self.args.max_steps - self.args.window_length + 2 - leak_length,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_ret = []
        epoch_loss = []
        epoch_weight = []
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_ret, batch_mask = self.get_episode_data(signal_matrix, ret_matrix, mask_matrix,
                                                                      batch_start_idx, leak_length=leak_length)
            input_mask = None
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                T, N, C = batch_data.shape
                batch_data = batch_data.reshape(T * N, C)
                scores = policy_net(batch_data, input_mask)
                if self.args.model == 'mlp':
                    # output: T x N x 1
                    scores = scores.reshape(T, N)
                    scores = F.relu(scores) + 1e-10
                    scores = torch.log(scores)

                if self.args.explore_noise and train:
                    noise = torch.randn(scores.shape).to(self.device)
                    scores = scores + noise * torch.std(scores) / epoch
                if self.args.weight_adj == 'raw':
                    weight = scores * batch_mask
                elif self.args.weight_adj == 'adj':
                    weight = scores * batch_mask
                    weight = functions.torch_get_adjusted_weight(weight)
                elif self.args.weight_adj == 'self':
                    # scores = torch.sigmoid(scores)
                    scores = scores * batch_mask
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                cross_ret = torch.sum(batch_ret * weight, dim=1)
                epoch_weight.append(weight.detach().cpu().numpy())
                eq_weight_ret = torch.mean(batch_ret * batch_mask, dim=1)
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

    def step1_train_leak_model(self):
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
            train_loss = self.train_episode_by_episode(self.train_data[0], self.train_data[1], self.train_data[2],
                                                       optimizer, self.model, scaler,
                                                       sample_factor=self.args.sample_factor, train=True)
            self.model.eval()
            with torch.no_grad():
                print('======valid=======')
                valid_loss, valid_weight = self.train_episode_by_episode(self.valid_data[0], self.valid_data[1],
                                                                         self.valid_data[2], optimizer, self.model,
                                                                         scaler,
                                                                         sample_factor=1, train=False, shuffle=False,
                                                                         epoch=epoch + 1)
                print('======test=======')
                test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1],
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
        # origin_policy = self.args.sample_policy
        self.args.sample_policy = 'random'
        self.model.eval()
        test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1], self.test_data[2],
                                                               optimizer,
                                                               self.model, scaler, train=False, shuffle=False,
                                                               sample_factor=1, return_weights=True)
        return test_weight

    def step2_get_leak_label(self, signal_matrix, ret_matrix, mask_matrix, policy_net,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1,
                                 leak_length=1):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[0]
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length + 2 - leak_length)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=0,
                                  stop=T - self.args.max_steps - self.args.window_length + 2 - leak_length,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_ret = []
        epoch_loss = []
        epoch_weight = []
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_ret, batch_mask = self.get_episode_data(signal_matrix, ret_matrix, mask_matrix,
                                                                      batch_start_idx, leak_length=1)
            input_mask = None
            T, N, C = batch_data.shape
            batch_data = batch_data.reshape(T * N, C)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                scores = policy_net(batch_data, input_mask)
                if self.args.model == 'mlp':
                    # output: T x N x 1
                    scores = scores.reshape(T, N)
                if self.args.explore_noise and train:
                    noise = torch.randn(scores.shape).to(self.device)
                    scores = scores + noise * torch.std(scores) / epoch
                if self.args.weight_adj == 'raw':
                    weight = scores * batch_mask
                elif self.args.weight_adj == 'adj':
                    weight = scores * batch_mask
                    weight = functions.torch_get_adjusted_weight(weight)
                elif self.args.weight_adj == 'self':
                    # scores = torch.sigmoid(scores)
                    scores = scores * batch_mask
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                cross_ret = torch.sum(batch_ret * weight, dim=1)
                epoch_weight.append(weight.detach().cpu().numpy())
                eq_weight_ret = torch.mean(batch_ret * batch_mask, dim=1)
                if self.args.target == 'sharpe':
                    gt = torch.mean(cross_ret) / torch.std(cross_ret)
                    eq_weight_sr = torch.mean(eq_weight_ret) / torch.std(eq_weight_ret)
                    if self.args.baseline:
                        gt -= eq_weight_sr
                elif self.args.target == 'ac_sharpe':
                    trading_cost = torch.sum(torch.cat([weight[:1], torch.abs(torch.diff(weight, dim=0))]) * 0.0015,
                                             dim=1)
                    ac_cross_ret = cross_ret - trading_cost
                    gt = torch.mean(ac_cross_ret) / torch.std(ac_cross_ret)
                if self.args.gradient_std == 'std':
                    loss = -torch.mean(gt) / self.args.batch_size
                else:
                    loss = -torch.mean(gt)
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

    def step3_train_leak_label(self, signal_matrix, label_matrix, mask_matrix, optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1,
                                 leak_length=0):
        # 原始矩阵 N x T x C, 对于conv, 得到T x C x N, 对于linear，得到T x N x C
        T = signal_matrix.shape[0]
        criterion = torch.nn.MSELoss(reduction='none')
        if self.args.sample_policy == 'random':
            start_idx = np.arange(0, T - self.args.max_steps - self.args.window_length + 2 - leak_length)
            sample_num = max(1, int(start_idx.shape[0] * sample_factor))
        elif self.args.sample_policy == 'fixed':
            start_idx = np.arange(start=0,
                                  stop=T - self.args.max_steps - self.args.window_length + 2 - leak_length,
                                  step=self.args.max_steps)
            sample_num = max(1, int(start_idx.shape[0]))
        if shuffle:
            np.random.shuffle(start_idx)
        epoch_loss = []
        for batch_idx in range(sample_num):
            batch_start_idx = start_idx[batch_idx]
            batch_data, batch_label, batch_mask = self.get_episode_data(signal_matrix, label_matrix, mask_matrix,
                                                                      batch_start_idx, leak_length=leak_length)
            input_mask = None
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                T, N, C = batch_data.shape
                batch_data = batch_data.reshape(T * N, C)
                # plt.figure()
                # plt.hist(batch_label[0].detach().cpu().numpy())
                # plt.show()
                batch_label = torch.exp(batch_label.reshape(T, N)) * batch_mask
                scores = policy_net(batch_data, input_mask).reshape(T, N)
                scores = F.relu(scores) + 1e-10
                weight = scores * batch_mask
                loss = criterion(weight, batch_label)
                loss = torch.mean(loss * torch.softmax(torch.square(batch_label), dim=1))
            if train:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.args.batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            epoch_loss.append(loss.detach().cpu().numpy())
        epoch_loss = np.mean(epoch_loss)
        print('loss', epoch_loss)
        return epoch_loss

    def train(self):
        leak_args = pkl.load(open(f"{self.target_data_path}/2015_config.pt", 'rb'))
        leak_model = MLP_linear(leak_args, out_dim=1).to(self.device)
        save_path = f"{self.target_data_path}/{self.args.test_start_date.split('-')[0]}_checkpoint.pt"
        leak_model.load_state_dict(torch.load(save_path))
        with torch.no_grad():
            leak_model.eval()
            train_loss, train_weight = self.step2_get_leak_label(self.train_data[0], self.train_data[1], self.train_data[2]
                                                    ,leak_model, sample_factor=self.args.sample_factor, train=False, return_weights=True)
            train_weight = torch.from_numpy(np.concatenate(train_weight, axis=0))
            valid_loss, valid_weight = self.step2_get_leak_label(self.valid_data[0], self.valid_data[1],
                                                                 self.valid_data[2]
                                                                 , leak_model, sample_factor=self.args.sample_factor,
                                                                 train=False, return_weights=True)
            valid_weight = torch.from_numpy(np.concatenate(valid_weight, axis=0))

        # train_dataset = CustomImageDataset(self.args, self.train_data[0], train_weight)
        # valid_dataset = CustomImageDataset(self.args, self.valid_data[0], valid_weight)
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True)
        # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True)
        policy_net = MLP_linear(self.args, out_dim=1).to(self.device)
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.args.lr)
        if self.args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3,
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
            policy_net.train()
            train_loss = self.step3_train_leak_label(self.train_data[0], train_weight, self.train_data[2], optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=True, epoch=1,
                                 leak_length=0)
            policy_net.eval()
            with torch.no_grad():
                print('======valid=======')
                valid_loss = self.step3_train_leak_label(self.valid_data[0], valid_weight, self.valid_data[2], optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=False, shuffle=True, return_weights=True, epoch=1,
                                 leak_length=0)
                print('======test=======')
                test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1],
                                                                       self.test_data[2], optimizer, policy_net, scaler,
                                                                       sample_factor=1, train=False, shuffle=False,
                                                                       epoch=epoch + 1, leak_length=0)
                if self.args.train_strategy == 'earlyStopping':
                    early_stopping(valid_loss, policy_net, self.save_path)
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
            torch.save(policy_net.state_dict(), self.save_path)
        policy_net.load_state_dict(torch.load(self.save_path))
        # origin_policy = self.args.sample_policy
        self.args.sample_policy = 'random'
        policy_net.eval()
        test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1], self.test_data[2],
                                                               optimizer,
                                                               policy_net, scaler, train=False, shuffle=False,
                                                               sample_factor=1, return_weights=True, leak_length=0)
        return test_weight

        #
