import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from models.embed import DataEmbedding
from models.ffn import MLP_Layer_2D

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, args, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(args.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input B x N x C, adj B x N x N, weight C x C
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        else:
            output = output
        return self.dropout(F.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)

        self.gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, adj, y_enc=None):
        # input: B x N x C

        x = self.enc_embedding(x)

        for gc in self.gcn:
            if self.args.gcn_res:
                x = x + gc(x, adj)
            else:
                x = gc(x, adj)
        x = self.projection(x)
        return x



class GCN_with_encoder(nn.Module):
    def __init__(self, args):
        super(GCN_with_encoder, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        self.encoder = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in range(self.args.num_enc_layers)])
        self.gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, adj, y_enc=None):
        # input: B x N x C
        B, N, C = x.shape
        x = x.reshape(-1, self.args.char_dim)
        x = self.enc_embedding(x)
        for layer in self.encoder:
            x = layer(x)
        x = x.reshape(B, N, -1)
        for gc in self.gcn:
            if self.args.gcn_res:
                x = x + gc(x, adj)
            else:
                x = gc(x, adj)
        x = self.projection(x)
        return x


class GCN_before_encoder(nn.Module):
    def __init__(self, args):
        super(GCN_before_encoder, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        self.encoder = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in range(self.args.num_enc_layers)])
        self.gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, adj, y_enc=None):
        # input: B x N x C
        B, N, C = x.shape
        x = self.enc_embedding(x)
        for gc in self.gcn:
            if self.args.gcn_res:
                x = x + gc(x, adj)
            else:
                x = gc(x, adj)
        x = x.reshape(-1, self.args.embed_dim)
        for layer in self.encoder:
            x = layer(x)
        x = x.reshape(B, N, -1)
        x = self.projection(x)
        return x

class GCN_2_encoder(nn.Module):
    def __init__(self, args):
        super(GCN_2_encoder, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        self.encoder1 = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in range(self.args.num_enc_layers)])
        self.encoder2 = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in
             range(self.args.num_enc_layers)])
        self.gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, adj, y_enc=None):
        # input: B x N x C
        B, N, C = x.shape
        x = x.reshape(-1, self.args.char_dim)
        x = self.enc_embedding(x)
        for layer in self.encoder1:
            x = layer(x)
        x = x.reshape(B, N, -1)
        for gc in self.gcn:
            if self.args.gcn_res:
                x = x + gc(x, adj)
            else:
                x = gc(x, adj)
        x = x.reshape(-1, self.args.embed_dim)
        for layer in self.encoder2:
            x = layer(x)
        x = x.reshape(B, N, -1)
        x = self.projection(x)
        return x