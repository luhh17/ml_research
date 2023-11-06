import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ffn import MLP_linear

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.asset_encoder = MLP_linear(args, out_dim=1)

    def forward(self, state):
        return self.asset_encoder(state)


class Critic(nn.Module):
    def __init__(self, args, asset_num):
        super(Critic, self).__init__()
        self.args = args
        self.asset_encoder = MLP_linear(args, out_dim=8)
        self.asset_scorer = MLP_linear(args=None, out_dim=2, d_model=64, input_dim=8+1, num_enc_layers=1)
        self.q_func = MLP_linear(args=None, out_dim=1, d_model=64, input_dim=asset_num*2, num_enc_layers=1)

    def forward(self, state, action):
        # B x N x C
        asset_char_vec = self.asset_encoder(state)
        asset_cur_vec = torch.cat([asset_char_vec, action], dim=-1)
        asset_vec = self.asset_scorer(asset_cur_vec)
        cross_section_vec = asset_vec.reshape(state.shape[0], -1)
        q_value = self.q_func(cross_section_vec)
        return q_value

