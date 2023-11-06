import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from models.rnn import RNN
from models.transformer import Transformer, Informer
from models.ffn import MLP_conv, MLP_linear
from utils.masking import generate_lookback_mask
from models.agents import Actor, Critic
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import pickle as pkl
from utils.early_stop import EarlyStopping
from prepare_dataset.myDataset import PortDataset, divide_train_valid_test, divide_slice_train_valid_test
from utils.loss_func import RetMSELoss
from utils.rl_env import TradeEnv, ReplayBuffer, Transition
'''
基础的训练环境，通过这个类调用train和valid，简化主函数
'''


class DDPG_Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._get_data()
        self.build_model()
        if self.args.est_method == 'rolling':
            self.save_path = f"{self.args.model_file_path}/{args.test_start_date.split('-')[0]}_checkpoint.pt"
        else:
            self.save_path = f'{self.args.model_file_path}/checkpoint.pt'
        self.memory = ReplayBuffer(args)
        self._select_optimizer()

    def save_args(self):
        if self.args.est_method == 'rolling':
            pkl.dump(self.args, open(f"{self.args.model_file_path}/{self.args.test_start_date.split('-')[0]}_config.pt", 'wb'))
        else:
            pkl.dump(self.args, open(f'{self.args.model_file_path}/config.pt', 'wb'))


    def _acquire_device(self):
        device_ids = self.args.devices.split(',')
        self.device_ids = [int(id_) for id_ in device_ids]
        device = torch.device('cuda')
        print('Use GPU: cuda:{}'.format(self.args.devices))
        return device

    def _get_data(self):
        test_year = self.args.test_start_date.split('-')[0]
        train, valid, test = pkl.load(open(f'/mnt/HDD16TB/huahao/portfolio_construction/data/{self.args.train_data_range}_{test_year}.pkl', 'rb'))
        # 原始矩阵 N x T x C or N x T
        self.train_env = TradeEnv(self.args, train)
        train = [np.transpose(t, (1, 0, 2)) if len(t.shape) == 3 else np.transpose(t, (1, 0)) for t in train[:3]]
        valid = [np.transpose(t, (1, 0, 2)) if len(t.shape) == 3 else np.transpose(t, (1, 0)) for t in valid[:3]]
        test = [np.transpose(t, (1, 0, 2)) if len(t.shape) == 3 else np.transpose(t, (1, 0)) for t in test[:3]]
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        # self.valid_env = TradeEnv(valid, self.args)
        # self.test_env = TradeEnv(test, self.args)

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
        elif self.args.optim == 'sgd':
            self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=self.args.lr)
            self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=self.args.lr)

    '''
    第一步，根据当前的policy，采样batch条episode
    '''
    def prob_to_weight(self, scores, mask, device):
        # scores 输入应当为 T x N
        # mask 输入应当为 T x N
        g_num = self.args.topk
        prob = torch.from_numpy(np.zeros((scores.shape[0], scores.shape[1]))).float().to(device)
        weights = torch.from_numpy(np.zeros((scores.shape[0], scores.shape[1]))).float().to(device)
        long_scores = scores * mask
        long_s, long_idx = torch.topk(long_scores, g_num, dim=1)
        long_ratio = torch.softmax(long_s, dim=1)
        short_scores = (1 - scores) * mask
        short_s, short_idx = torch.topk(short_scores, g_num, dim=1)
        short_ratio = torch.softmax(short_s, dim=1)
        for i, indice in enumerate(long_idx):
            weights[i, indice] = long_ratio[i]
            prob[i, indice] = long_ratio[i]
        for i, indice in enumerate(short_idx):
            weights[i, indice] = -short_ratio[i]
            prob[i, indice] = short_ratio[i]
        prob = prob.contiguous()
        weights = weights.contiguous()
        return prob, weights

    def get_episode_data(self, signal_matrix, ret_matrix, mask_matrix, batch_start_idx):
        # T x C x N or T x N x C
        batch_ret = ret_matrix[batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_ret = batch_ret.to(self.device)
        batch_data = signal_matrix[batch_start_idx: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_data = batch_data.to(self.device)
        batch_mask = mask_matrix[batch_start_idx + self.args.window_length - 1: batch_start_idx + self.args.window_length + self.args.max_steps - 1]
        batch_mask = batch_mask.to(self.device)
        return batch_data, batch_ret, batch_mask

    def soft_update(self, target, source, tau=0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_param(self, actor_scaler, critic_scaler):
        transitions = self.memory.sample(self.args.batch_size)
        batch = Transition(*zip(*transitions))
        gamma = 1
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        mask_batch = torch.stack(batch.mask)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        state_batch = state_batch.squeeze(-2)
        action_batch = action_batch.squeeze(-2)
        next_state_batch = next_state_batch.squeeze(-2)

        q_batch = self.critic(state_batch, action_batch)
        next_action_batch = self.actor_target(next_state_batch)
        next_q_batch = reward_batch + gamma * self.critic_target(next_state_batch, next_action_batch)

        value_loss = F.mse_loss(q_batch, next_q_batch)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        policy_loss = - self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        self.soft_update(self.actor_target, self.actor, tau=0.001)
        self.soft_update(self.critic_target, self.critic, tau=0.01)
        return value_loss, policy_loss


    def train_episode_by_episode(self, signal_matrix, ret_matrix, mask_matrix, optimizer, policy_net, scaler,
                                 sample_factor=0.1, train=True, shuffle=True, return_weights=False):
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
            batch_data, batch_ret, batch_mask = self.get_episode_data(signal_matrix, ret_matrix, mask_matrix,
                                                                      batch_start_idx)
            input_mask = None
            if self.args.model != 'mlp':
                input_mask = generate_lookback_mask(batch_data.shape[0], self.args.window_length).to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                scores = policy_net(batch_data)
                if self.args.mlp_implement == 'linear':
                    # output: T x N x 1
                    scores = scores.squeeze(dim=2)[self.args.window_length - 1:]
                elif self.args.mlp_implement == 'conv':
                    # output: T x 1 x N
                    scores = scores.squeeze(dim=1)[:, self.args.window_length - 1:]
                if self.args.weight_adj == 'raw':
                    weight = scores * batch_mask
                elif self.args.weight_adj == 'self':
                    scores = torch.sigmoid(scores)
                    prob, weight = self.prob_to_weight(scores, batch_mask, self.device)
                cross_ret = torch.sum(batch_ret * weight, dim=1)
                epoch_weight.append(weight.detach().cpu().numpy())

                gt = torch.mean(cross_ret) / torch.std(cross_ret)
                if self.args.target == 'logret':
                    log_ret = torch.log(cross_ret + 1)
                    gt = 0
                    for i in range(cross_ret.shape[0]):
                        gt += log_ret[i] * (self.args.gamma ** i)
                # gt = (gt - torch.mean(gt)) / (torch.std(gt) + 1e-9)
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

    def train(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optim, mode='min', factor=0.2, patience=3, verbose=True)
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optim, mode='min', factor=0.2, patience=3, verbose=True)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(self.args.epochs):
            print('epoch', epoch + 1)
            print('======train=======')
            start_time = time.time()

            self.actor_target.train()
            self.actor.train()
            self.critic.train()
            self.critic_target.train()

            state, mask = self.train_env.reset()
            state = state.to(self.device)
            mask = mask
            value_loss_list = []
            policy_loss_list = []
            for t in range(100):
                with torch.no_grad():
                    action = self.actor(state)
                    next_state, reward, done = self.train_env.step(action.detach().cpu())  # env.step() takes numpy array as inputs
                    self.memory.push(state.detach().cpu(), action.detach().cpu(), mask, reward, next_state)
                if len(self.memory) >= self.args.batch_size:
                    for _ in range(2):
                        value_loss, policy_loss = self.update_param(scaler, scaler)
                        value_loss_list.append(value_loss.item())
                        policy_loss_list.append(policy_loss.item())
                state = next_state.to(self.device)
                if done:
                    break
            print('value loss', np.mean(value_loss_list))
            print('policy_loss', np.mean(policy_loss_list))

            self.actor_target.eval()
            self.actor.eval()
            self.critic.eval()
            self.critic_target.eval()
            with torch.no_grad():
                print('======valid=======')
                self.args.sample_policy = 'random'
                valid_loss = self.train_episode_by_episode(self.valid_data[0], self.valid_data[1], self.valid_data[2],
                                                           None, self.actor_target, scaler,
                                                           sample_factor=1, train=False)
                if self.args.train_strategy == 'earlyStopping':
                    early_stopping(valid_loss, self.actor_target, self.save_path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                critic_scheduler.step(valid_loss)
                actor_scheduler.step(valid_loss)


            end_time = time.time()
            print('epoch time {0:.2f}'.format(end_time - start_time))
        if self.args.train_strategy == 'full':
            torch.save(self.actor_target.state_dict(), self.save_path)
        self.actor_target.load_state_dict(torch.load(self.save_path))
        # origin_policy = self.args.sample_policy
        self.args.sample_policy = 'random'
        test_loss, test_weight = self.train_episode_by_episode(self.test_data[0], self.test_data[1], self.test_data[2],
                                                               None,
                                                               self.actor_target, scaler, train=False, shuffle=False,
                                                               sample_factor=1, return_weights=True)
        return test_weight

    def build_model(self):
        asset_num = self.train_data[0].shape[1]
        self.critic = Critic(self.args, asset_num).to(self.device)
        self.critic_target = Critic(self.args, asset_num).to(self.device)
        self.actor = Actor(self.args).to(self.device)
        self.actor_target = Actor(self.args).to(self.device)
        if len(self.device_ids) > 1:
            self.critic = torch.nn.DataParallel(self.critic)
            self.critic_target = torch.nn.DataParallel(self.critic_target)
            self.actor = torch.nn.DataParallel(self.actor)
            self.actor_target = torch.nn.DataParallel(self.actor_target)

