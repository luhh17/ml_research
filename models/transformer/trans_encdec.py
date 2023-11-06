import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.self_atten import FullAttention, AttentionLayer, CausalAttention
import copy

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_input, d_hidden=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_hidden = d_hidden or 4*d_input
        # attention 可以是Full-Attention或是Sparse Attention
        self.attention = attention
        # 1x1 conv 等价于 Linear
        # self.conv1 = nn.Conv1d(in_channels=d_input, out_channels=d_hidden, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_hidden, out_channels=d_input, kernel_size=1)
        self.ff1 = nn.Linear(d_input, d_hidden)
        self.ff2 = nn.Linear(d_hidden, d_input)
        self.norm1 = nn.LayerNorm(d_input)
        self.norm2 = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, target=None, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))

        new_x, attn = self.attention(x, x, x, attn_mask=None)

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.ff1(y)))

        y = self.dropout(self.ff2(y))

        return self.norm2(x+y), target


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        # self.norm2 = copy.deepcopy(norm_layer)

    def forward(self, x, target=None, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            # attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, target = attn_layer(x, target=target, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x, target
    


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.ff1(y)))
        y = self.dropout(self.ff2(y))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        return x
