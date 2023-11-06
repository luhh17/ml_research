import torch
import random
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'mask', 'reward', 'next_state'))
trading_cost = 0.0015


class TradeEnv():
    def __init__(self, args, data_list):
        self.args = args
        # 原始矩阵 N x T x C or N x T
        self.signal_matrix = data_list[0]
        self.return_matrix = data_list[1]
        self.mask_matrix = data_list[2]
        self.num_assets = self.signal_matrix.shape[0]
        self.current_holding = torch.zeros(self.num_assets)
        self.current_idx = 0
        self.current_step = 0

    def step(self, action, trading_cost=0.0015):
        # today
        last_holding = self.current_holding
        ret_matrix = self.return_matrix[:, self.current_idx]
        mask = self.mask_matrix[:, self.current_idx]
        action = action.reshape(-1)
        ret = torch.sum(ret_matrix * action * mask)
        trading_cost = torch.sum(trading_cost * torch.abs(action - last_holding))
        self.current_holding = action
        self.current_idx += 1
        self.current_step += 1
        self.current_holding = action
        next_state = self.signal_matrix[:, self.current_idx:self.current_idx + self.args.window_length, :]
        if (self.current_step >= self.args.max_steps) or (self.current_idx >= self.return_matrix.shape[1]):
            done = True
        else:
            done = False

        # print(next_state.shape)
        # print(self.signal_matrix.shape)
        # print(self.current_idx)
        # print(self.args.window_length)
        if self.args.target == 'bc_return':
            reward = ret
        elif self.args.target == 'ac_return':
            reward = ret - trading_cost
        elif self.args.target == 'bc_sharpe':
            raise NotImplementedError
        elif self.args.target == 'ac_sharpe':
            raise NotImplementedError
        return next_state, reward, done

    def reset(self, idx=None):
        if idx is None:
            self.current_idx = random.choice(range(self.return_matrix.shape[1] - self.args.window_length - 1))
        else:
            self.current_idx = idx
        self.current_step = 0
        self.current_holding = torch.zeros(self.num_assets)
        init_state = self.signal_matrix[:, self.current_idx:self.current_idx + self.args.window_length, :]
        init_mask = self.signal_matrix[:, self.current_idx:self.current_idx + self.args.window_length]
        return init_state, init_mask


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.capacity = self.args.capacity
        self.memory = []
        self.idx = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.idx] = Transition(*args)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)