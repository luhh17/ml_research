import torch
import torch.nn as nn
from utils.masking import TriangularCausalMask
from models.embed import DataEmbedding
from utils.data_organization import build_decoder_input
import torch.nn.functional as F
import pdb

class Conv_MLP_Layer(nn.Module):
    def __init__(self, args, d_input, d_hidden=None, dropout=0.1, activation="relu"):
        super(Conv_MLP_Layer, self).__init__()
        self.args = args
        d_hidden = d_hidden or 4*d_input
        # 1x1 conv 等价于 Linear
        self.conv1 = nn.Conv1d(in_channels=d_input, out_channels=d_hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_hidden, out_channels=d_input, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_input)
        self.norm2 = nn.BatchNorm1d(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        if self.args.mlp_type == 'res':
            return self.norm2(x+y)
        elif self.args.mlp_type == 'vanilla':
            return self.norm2(y)


class MLP_Layer(nn.Module):
    def __init__(self, args, d_input, d_hidden=None, dropout=0.1, activation="relu", mlp_type='res', typical=False):
        super(MLP_Layer, self).__init__()
        d_hidden = d_hidden or 4*d_input
        # 1x1 conv 等价于 Linear
        self.args = args
        self.linear1 = nn.Linear(in_features=d_input, out_features=d_hidden)
        self.linear2 = nn.Linear(in_features=d_hidden, out_features=d_input)
        self.mlp_type = self.args.mlp_type if self.args else mlp_type
        self.norm1 = nn.BatchNorm1d(d_input)
        self.norm2 = nn.BatchNorm1d(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.typical = typical

    def forward(self, x):
        # x [B, L, D

        x = x.transpose(-1, 1)
        y = self.norm1(x).transpose(-1, 1)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y)).transpose(-1, 1)
        if self.mlp_type == 'res':
            return self.norm2(x + y).transpose(-1, 1)
        elif self.mlp_type == 'vanilla':
            return self.norm2(y).transpose(-1, 1)
        #
        # y = self.norm1(x)
        # y = self.dropout(self.activation(self.linear1(y)))
        # y = self.dropout(self.linear2(y))
        # if self.mlp_type == 'res':
        #     return self.norm2(x + y)
        # elif self.mlp_type == 'vanilla':
        #     return self.norm2(y)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        d_model = args.embed_dim
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        # self.enc_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
        # self.enc_embedding = nn.Linear(self.args.char_dim, d_model)
        self.enc_embedding = nn.Conv1d(in_channels=self.args.char_dim, out_channels=d_model, kernel_size=1)
        # Encoder
        if args.mlp_implement == "conv":
            self.encoder = nn.ModuleList([Conv_MLP_Layer(args, d_model, d_ff, dropout=dropout) for i in range(args.num_enc_layers)])
        elif args.mlp_implement == 'linear':
            self.encoder = nn.ModuleList([MLP_Layer(args, d_model, d_ff, dropout=dropout) for i in range(args.num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_enc):
        x_enc = x_enc.transpose(-1, 1)
        enc_out = self.enc_embedding(x_enc).transpose(-1, 1)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out)
        return dec_out


class serial_MLP(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=True):
        super(serial_MLP, self).__init__()
        d_ff = 4 * d_model
        if args is not None:
            self.args = args
            input_dim = self.args.char_dim
            d_model = args.embed_dim
            d_ff = args.hidden_dim
            dropout = args.dropout
            num_enc_layers = args.num_enc_layers

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList([MLP_Layer(args, d_model+1, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model+1, out_dim, bias=True)

    def forward(self, x_enc, last_weight):
        # N x C, N x 1
        enc_out = self.enc_embedding(x_enc)
        enc_out = torch.cat([enc_out, last_weight], dim=-1)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out)
        return dec_out


class MLP_Layer_2D(nn.Module):
    def __init__(self, args, d_input, d_hidden=None, dropout=0.1, activation="relu", mlp_type='res'):
        super(MLP_Layer_2D, self).__init__()
        d_hidden = d_hidden or 4*d_input
        # 1x1 conv 等价于 Linear
        self.args = args
        self.linear1 = nn.Linear(in_features=d_input, out_features=d_hidden)
        self.linear2 = nn.Linear(in_features=d_hidden, out_features=d_input)
        self.mlp_type = self.args.mlp_type if self.args else mlp_type
        self.norm1 = nn.BatchNorm1d(d_input)
        self.norm2 = nn.BatchNorm1d(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x [B, D]

        y = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y)

class MLP_linear(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(MLP_linear, self).__init__()
        # input B x L x D
        d_ff = 4 * d_model
        if args is not None:
            self.args = args
            input_dim = self.args.char_dim
            d_model = args.embed_dim
            d_ff = args.hidden_dim
            dropout = args.dropout
            num_enc_layers = args.num_enc_layers

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList([MLP_Layer(args, d_model, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, x_enc, input_mask=None, y_enc=None):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out)
        # dec_out = F.relu(dec_out) + 1e-10
        return dec_out[:, -1, :]
    
class MLP_linear_concat(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(MLP_linear_concat, self).__init__()
        # input B x L x D
        d_ff = 4 * d_model
        if args is not None:
            self.args = args
            input_dim_1 = self.args.basic_fea_num
            input_dim_2 = self.args.intra_fea_num

            d_model = args.embed_dim
            d_ff = args.hidden_dim
            dropout = args.dropout
            num_enc_layers = args.num_enc_layers

        # Encoding
        self.enc_embedding_1 = nn.Linear(input_dim_1, d_model)
        self.enc_embedding_2 = nn.Linear(input_dim_2, d_model)

        # Encoder
        self.encoder_1 = nn.ModuleList([MLP_Layer(args, d_model, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        self.encoder_2 = nn.ModuleList([MLP_Layer(args, d_model, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(2*d_model, out_dim, bias=True)

    def forward(self, x_enc, input_mask=None, y_enc=None):
        
        x_enc_1=x_enc[:,:,:self.args.basic_fea_num]
        x_enc_2=x_enc[:,:,-self.args.intra_fea_num:]

        enc_out_1 = self.enc_embedding_1(x_enc_1)
        enc_out_2 = self.enc_embedding_2(x_enc_2)

        for layer in self.encoder_1:
            enc_out_1 = layer(enc_out_1)
        for layer in self.encoder_2:
            enc_out_2 = layer(enc_out_2)
        
        enc_out=torch.cat([enc_out_1, enc_out_2], dim=-1)
        
        dec_out = self.projection(enc_out)
        # dec_out = F.relu(dec_out) + 1e-10
        return dec_out[:, -1, :]
    

class MLP_ts(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(MLP_ts, self).__init__()
        # input B x L x D
        d_ff = 4 * d_model
        if args is not None:
            self.args = args
            input_dim = self.args.char_dim
            d_model = args.embed_dim
            d_ff = args.hidden_dim
            dropout = args.dropout
            num_enc_layers = args.num_enc_layers

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList([MLP_Layer(args, d_model, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model*self.args.window_length, out_dim, bias=True)

    def forward(self, x_enc, input_mask=None, y_enc=None):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        dec_out = self.projection(enc_out)
        # dec_out = F.relu(dec_out) + 1e-10
        return dec_out


class MLP_post_ts(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(MLP_post_ts, self).__init__()
        # input B x L x D
        d_ff = 4 * d_model
        if args is not None:
            self.args = args
            input_dim = self.args.char_dim
            d_model = args.embed_dim
            d_ff = args.hidden_dim
            dropout = args.dropout
            num_enc_layers = args.num_enc_layers

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList([MLP_Layer(args, d_model, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model, out_dim, bias=True)

        self.agg_timeStep = nn.Linear(self.args.window_length, 1, bias=True)

    def forward(self, x_enc, input_mask=None, y_series=None):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out).reshape(enc_out.shape[0], -1)
        dec_out = self.agg_timeStep(dec_out)
        # dec_out = F.relu(dec_out) + 1e-10
        return dec_out


class MLP_RNN(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(MLP_RNN, self).__init__()
        # input B x L x D
        d_ff = 4 * d_model
        if args is not None:
            self.args = args
            input_dim = self.args.char_dim
            d_model = args.embed_dim
            d_ff = args.hidden_dim
            dropout = args.dropout
            num_enc_layers = args.num_enc_layers

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList(
            [MLP_Layer(args, d_model, d_ff, dropout=dropout, typical=typical) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model, out_dim, bias=True)

        self.rnn_encoder = nn.RNN(input_size=d_model, hidden_size=d_model,
                                  num_layers=2, batch_first=True)

    def forward(self, x_enc, input_mask=None, y_series=None):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        enc_out, h_n = self.rnn_encoder(enc_out, None)
        dec_out = self.projection(enc_out[:, -1, :]).reshape(-1, 1)
        return dec_out


class MLP_conv(nn.Module):
    def __init__(self, args):
        # input: B x D x L,
        super(MLP_conv, self).__init__()
        self.args = args
        d_model = args.embed_dim
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        self.enc_embedding = nn.Conv1d(in_channels=self.args.char_dim, out_channels=d_model, kernel_size=1)
        # Encoder
        self.encoder = nn.ModuleList([Conv_MLP_Layer(args, d_model, d_ff, dropout=dropout) for i in range(args.num_enc_layers)])
        # Decoder
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1)

    def forward(self, x_enc, input_mask=None):
        x_enc = x_enc
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out)
        return dec_out


class Diff_gap_model(nn.Module):
    def __init__(self, args, pred_net_list):
        # input: B x D x L,
        super(Diff_gap_model, self).__init__()
        self.args = args
        self.pred_net_list = pred_net_list

    def forward(self, batch_data, input_mask=None):
        scores_list = []
        for pred_net in self.pred_net_list:
            scores = pred_net(batch_data, input_mask)[self.args.window_length - 1:]
            scores_list.append(scores)
        return torch.cat(scores_list, dim=-1)
