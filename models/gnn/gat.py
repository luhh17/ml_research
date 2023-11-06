import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embed import DataEmbedding
from models.ffn import MLP_Layer_2D


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # input B x N x C, adj B x N x N, weight C x C
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class GAT(nn.Module):
    def __init__(self, args, nheads, alpha=0.01):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.args = args
        nfeat = self.args.embed_dim
        hidden_dim = self.args.embed_dim
        self.dropout = args.dropout
        nhid = hidden_dim // nheads

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=self.dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(hidden_dim, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
    


class GAT_with_encoder(nn.Module):
    def __init__(self, args):
        super(GAT_with_encoder, self).__init__()
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
                GAT(args, args.num_heads) for _ in range(args.num_enc_layers)
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