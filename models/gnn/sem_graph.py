import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from models.embed import DataEmbedding
from models.ffn import MLP_Layer_2D
from models.gnn.gcn import GraphConvolution
from models.layers.self_atten import AttentionLayer, FullAttention
from utils.graph_matrix import norm_adj, dummy_adj

class Sem_Graph(nn.Module):
    def __init__(self, args):
        super(Sem_Graph, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        # self_atten = FullAttention(attention_dropout=self.args.dropout, output_attention=True)
        # self.self_atten = AttentionLayer(self_atten, d_model, args.num_heads)
        self.inner_attention = FullAttention(attention_dropout=self.args.dropout, output_attention=True)
        self.encoder = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in range(self.args.num_enc_layers)])
      
        self.gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, adj=None, y_enc=None, atten_mask=None):
        # input: B x N x C
        # for our data, true for mask means the stock exists
        # but for MHA, true for mask means the stock does not exist
        atten_mask = ~(atten_mask).unsqueeze(1)
        x = self.enc_embedding(x)
        B, L, D = x.shape
        x = x.reshape(-1, D)
        for layer in self.encoder:
            x = layer(x)
        x = x.reshape(B, L, D)
        # _, adj_mat = self.self_atten(x, x, x, attn_mask=atten_mask)
        B, L, D = x.shape
        H = self.args.num_heads
        out, adj_mat = self.inner_attention(x.view(B, L, H, -1), x.view(B, L, H, -1), x.view(B, L, H, -1), attn_mask=atten_mask)
        b, l, h, d = out.shape
        # out : [Batch_size, Query_len, Head_num, Value_dim]
        out = out.view(b, l, -1)
        adj_mat = torch.mean(adj_mat, 1)

        # print(torch.sum(torch.isnan(adj_mat)))
        for gc in self.gcn:
            if self.args.use_atten:
                x = out
                break
            if self.args.gcn_res:
                x = x + gc(x, adj_mat)
            else:
                x = gc(x, adj_mat)
        x = self.projection(x)
        return x, adj_mat


class Sem_Graph_v2(nn.Module):
    def __init__(self, args):
        super(Sem_Graph_v2, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        # self_atten = FullAttention(attention_dropout=self.args.dropout, output_attention=True)
        # self.self_atten = AttentionLayer(self_atten, d_model, args.num_heads)
        self.encoder = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in range(self.args.num_enc_layers)])
      

        self.inner_attention = nn.ModuleList(
            [
                FullAttention(attention_dropout=self.args.dropout, output_attention=True) for _ in range(args.num_enc_layers)
            ]
        )
        
        self.gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )

        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, adj=None, y_enc=None, atten_mask=None):
        # input: B x N x C
        # for our data, true for mask means the stock exists
        # but for MHA, true for mask means the stock does not exist
        
        atten_mask = ~(atten_mask).unsqueeze(1)
        x = self.enc_embedding(x)
        B, L, D = x.shape
        x = x.reshape(-1, D)
        for layer in self.encoder:
            x = layer(x)
        x = x.reshape(B, L, D)
        # _, adj_mat = self.self_atten(x, x, x, attn_mask=atten_mask)
        
        H = self.args.num_heads
        for atten, gcn in zip(self.inner_attention, self.gcn):
            _, adj_mat = atten(x.view(B, L, H, -1), x.view(B, L, H, -1), x.view(B, L, H, -1), attn_mask=atten_mask)
            b, l, h, d = _.shape
            _ = _.view(b, l, -1)
            adj_mat = torch.mean(adj_mat, 1)
            if self.args.adj_norm:
                adj_mat = norm_adj(adj_mat, self.args.add_eye)
            if self.args.gcn_res:
                x = x + gcn(x, adj_mat)
            else:
                x = gcn(x, adj_mat)

        x = self.projection(x)
        return x, adj_mat

