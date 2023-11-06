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
from models.gnn.sem_graph import Sem_Graph_v2
from models.gnn.gcn import GraphConvolution



class Dual(nn.Module):
    def __init__(self, args):
        super(Dual, self).__init__()
        self.args = args
        d_model = args.embed_dim
        if self.args.only_ret:
            self.enc_embedding = nn.Linear(1, d_model)
        else:
            self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        self.encoder = nn.ModuleList(
            [MLP_Layer_2D(args, d_model, d_model * 4, dropout=self.args.dropout) for i in range(self.args.num_enc_layers)])
      
        self.inner_attention = nn.ModuleList(
            [
                FullAttention(attention_dropout=self.args.dropout, output_attention=True) for _ in range(args.num_enc_layers)
            ]
        )
        
        self.sem_gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )

        self.syn_gcn = nn.ModuleList(
            [
                GraphConvolution(args, d_model, d_model) for _ in range(args.num_enc_layers)
            ]
        )

        self.projection = nn.Linear(d_model * 2, 1, bias=True)

    def forward(self, x, syn_adj=None, y_enc=None, atten_mask=None):
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
        H = self.args.num_heads
        # Sem GCN
        raw_x = x.clone()
        for atten, gcn in zip(self.inner_attention, self.sem_gcn):
            
            _, adj_mat = atten(x.view(B, L, H, -1), x.view(B, L, H, -1), x.view(B, L, H, -1), attn_mask=atten_mask)
            b, l, h, d = _.shape
            _ = _.view(b, l, -1)
            adj_mat = torch.mean(adj_mat, 1)
            # if self.args.adj_norm:
            #     adj_mat = norm_adj(adj_mat, self.args.add_eye)
            if self.args.gcn_res:
                x = x + gcn(x, adj_mat)
            else:
                x = gcn(x, adj_mat)

        # Syn GCN
        for gc in self.syn_gcn:
            if self.args.gcn_res:
                raw_x = raw_x + gc(raw_x, syn_adj)
            else:
                raw_x = gc(raw_x, syn_adj)
        
        hidden = torch.cat([x, raw_x], dim=-1)
        x = self.projection(hidden)
        return x, adj_mat

      
