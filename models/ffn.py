import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class MlpLayer3D(nn.Module):
    def __init__(self, args, d_input, d_hidden=None, dropout=0.1, activation="relu"):
        super(MlpLayer3D, self).__init__()
        d_hidden = d_hidden or 4 * d_input
        self.args = args
        self.linear1 = nn.Linear(in_features=d_input, out_features=d_hidden)
        self.linear2 = nn.Linear(in_features=d_hidden, out_features=d_input)
        self.norm1 = nn.BatchNorm1d(d_input)
        self.norm2 = nn.BatchNorm1d(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x [B, L, D]
        x = x.transpose(-1, 1)
        y = self.norm1(x).transpose(-1, 1)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y)).transpose(-1, 1)
        return self.norm2(x + y).transpose(-1, 1)


class MlpLayer2D(nn.Module):
    def __init__(self, args, d_input, d_hidden=None, dropout=0.1, activation="relu"):
        super(MlpLayer2D, self).__init__()
        d_hidden = d_hidden or 4 * d_input
        # 1x1 conv 等价于 Linear
        self.args = args
        self.linear1 = nn.Linear(in_features=d_input, out_features=d_hidden)
        self.linear2 = nn.Linear(in_features=d_hidden, out_features=d_input)
        self.norm1 = nn.BatchNorm1d(d_input)
        self.norm2 = nn.BatchNorm1d(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        y = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y)


class Mlp3D(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(Mlp3D, self).__init__()
        # input B x L x D
        d_ff = 4 * d_model
        self.args = args
        input_dim = self.args['char_dim']
        d_model = args['embed_dim']
        d_ff = args['hidden_dim'] or d_ff
        dropout = args['dropout']
        num_enc_layers = args['num_enc_layers']
        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList([MlpLayer3D(args, d_model, d_ff, dropout=dropout) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, x_enc, input_mask=None, y_enc=None):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out)
        return dec_out[:, -1, :]


class Mlp2D(nn.Module):
    def __init__(self, args, out_dim=1, d_model=512, input_dim=1024, num_enc_layers=6, dropout=0.1, typical=False):
        super(Mlp2D, self).__init__()
        # input B x D
        d_ff = 4 * d_model
        self.args = args
        input_dim = self.args['char_dim']
        d_model = args['embed_dim']
        d_ff = args['hidden_dim'] or d_ff
        dropout = args['dropout']
        num_enc_layers = args['num_enc_layers']

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        # Encoder
        self.encoder = nn.ModuleList([MlpLayer2D(args, d_model, d_ff, dropout=dropout) for i in range(num_enc_layers)])
        # Decoder
        self.projection = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, x_enc, input_mask=None, y_enc=None):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
        dec_out = self.projection(enc_out)
        return dec_out


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
  