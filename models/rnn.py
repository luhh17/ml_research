import torch
import torch.nn as nn
from utils.data_organization import build_decoder_input


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.embedding = nn.Linear(args.char_dim, args.embed_dim)
        self.rnn_encoder = nn.RNN(input_size=args.embed_dim, hidden_size=args.embed_dim, num_layers=2, batch_first=True)
        # self.rnn_decoder = nn.RNN(input_size=args.embed_dim, hidden_size=args.embed_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(args.embed_dim, 1)

    def forward(self, x, y_enc):
        x = self.embedding(x)
        '''
        (N, L, D*H), (D*num_layers, N, H)
        '''
        # print(x.shape)
        enc_out, h_n = self.rnn_encoder(x, None)
        # start_token = torch.zeros(x.shape[0], 1, x.shape[2]).to(self.args.device)
        # print(start_token.shape)
        # print(x.shape)
        # dec_out, h_n = self.rnn_decoder(start_token, h_n)
        out = self.out(enc_out[:, -1, :]).reshape(-1, 1)
        return out
    
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.embedding = nn.Linear(args.char_dim, args.embed_dim)
        self.lstm_encoder = nn.LSTM(input_size=args.embed_dim, hidden_size=args.embed_dim, num_layers=2, batch_first=True)
        # self.rnn_decoder = nn.RNN(input_size=args.embed_dim, hidden_size=args.embed_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(args.embed_dim, 1)

    def forward(self, x, y_enc):
        x = self.embedding(x)
        '''
        (N, L, D*H), (D*num_layers, N, H)
        '''
        # print(x.shape)
        enc_out, h_n = self.lstm_encoder(x, None)
        # start_token = torch.zeros(x.shape[0], 1, x.shape[2]).to(self.args.device)
        # print(start_token.shape)
        # print(x.shape)
        # dec_out, h_n = self.rnn_decoder(start_token, h_n)
        out = self.out(enc_out[:, -1, :]).reshape(-1, 1)
        return out