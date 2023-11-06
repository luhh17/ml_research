import torch
import math

torch.manual_seed(0)
#=================================================================================#
#=============================== Model Structure =================================#
#=================================================================================#

# Core modules

class FFNModel(torch.nn.Module):
    'A simple feed-forward neural network (FFN)'
    def __init__(self, d_input, d_output, d_hidden, activation=torch.nn.ReLU(), dropout=0.1, if_long=False, long_act=torch.nn.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'FFN'
        self.activation = activation
        try:
            self.n_layers = len(d_hidden)
        except TypeError:
            self.n_layers = 1
            d_hidden = [d_hidden]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(d_input, d_hidden[0])])
        for i in range(self.n_layers-1):
            self.layers.append(torch.nn.Linear(d_hidden[i], d_hidden[i+1]))
        self.layers.append(torch.nn.Linear(d_hidden[-1], d_output))
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = long_act
        self.if_long = if_long

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        if self.if_long:
            x = self.last_act(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        x = self.dropout(x)
        return x

class ResNetModel(torch.nn.Module):
    'A residual-structure neural network (ResNet)'
    def __init__(self, d_input, d_output, d_hidden, activation=torch.nn.ReLU(), dropout=0.1, if_long=False, long_act=torch.nn.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'ResNet'
        self.activation = activation
        try:
            self.n_layers = len(d_hidden)
        except TypeError:
            self.n_layers = 1
            d_hidden = [d_hidden]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(d_input, d_hidden[0])])
        for i in range(self.n_layers-1):
            self.layers.append(torch.nn.Linear(sum(d_hidden[:i+1]), d_hidden[i+1]))
        self.layers.append(torch.nn.Linear(sum(d_hidden), d_output))
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = long_act
        self.if_long = if_long

    def forward(self, x):
        for n, layer in enumerate(self.layers[:-1]):
            x1 = self.activation(layer(x))
            x1 = self.dropout(x1)
            if n == 0:
                x = x1
            else:
                x = torch.concat([x, x1], dim=-1)
        if self.if_long:
            x = self.last_act(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        x = self.dropout(x)
        return x

class ResMultiNetModel(torch.nn.Module):
    'A residual-structure & multiplication among different layers neural network (ResNet)'
    def __init__(self, d_input, d_output, d_hidden, activation=torch.nn.ReLU(), dropout=0.1, if_long=False, long_act=torch.nn.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'ResMulti'
        self.activation = activation
        try:
            self.n_layers = len(d_hidden)
        except TypeError:
            self.n_layers = 1
            d_hidden = [d_hidden]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(d_input, d_hidden[0])])
        for i in range(self.n_layers-1):
            self.layers.append(torch.nn.Linear(sum(d_hidden[:i+1]), d_hidden[i+1]))
        self.layers.append(torch.nn.Linear(sum(d_hidden), d_output))
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = long_act
        self.if_long = if_long

    def forward(self, x):
        x0 = self.activation(self.layers[0](x))
        x0 = self.dropout(x0)
        x = x0
        for n, layer in enumerate(self.layers[1:-1]):
            x1 = self.activation(layer(x))
            x1 = self.dropout(x1)
            x1 = torch.multiply(x1, x0) + x1
            x = torch.concat([x, x1], dim=-1)
        if self.if_long:
            x = self.last_act(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        x = self.dropout(x)
        return x

class WideDeepModel(torch.nn.Module):
    'A wide&deep structure neural network'
    '''
    d_hidden = [1024,512,256,128,64,32] which means [[1024] wide,[512,256,128,64,32] deep]
    '''
    'A simple feed-forward neural network (FFN)'
    
    def __init__(self, d_input, d_output, d_hidden, activation=torch.nn.ReLU(), dropout=0.1, if_long=False, long_act=torch.nn.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'FFN'
        self.activation = activation
        try:
            self.n_layers = len(d_hidden)
        except TypeError:
            self.n_layers = 1
            d_hidden = [d_hidden]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(d_input, d_hidden[0])])
        for i in range(self.n_layers-1):
            self.layers.append(torch.nn.Linear(d_hidden[i], d_hidden[i+1]))
        self.layers.append(torch.nn.Linear(d_hidden[-1], d_output))
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = long_act
        self.if_long = if_long

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        if self.if_long:
            x = self.last_act(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        x = self.dropout(x)
        return x

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000, batch_first=False):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            # Transfer to [T,N,K]
            x = x.transpose(0, 1)
            x = x + self.pe[:x.size(0)]
            x = x.transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(torch.nn.Module):

    def __init__(self, d_input, d_output, nhead, d_hidden, nlayers, dropout=0.1, batch_first=False, if_long=False, long_act=torch.nn.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_input, dropout, batch_first=batch_first)
        #self.pos_encoder = torch.nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=batch_first)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_input, nhead, d_hidden, dropout=dropout, batch_first=batch_first)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_input
        self.decoder = torch.nn.Linear(d_input, d_output)
        self.last_act = long_act
        self.if_long = if_long
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, d_model]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_output]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        if self.if_long:
            output = self.last_act(output)

        return output

#=================================================================================#
#==================================== Others ======================================#
#=================================================================================#

def generate_square_subsequent_mask(sz):
    'Generates an upper-triangular matrix of -inf, with zeros on diag.'
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def generate_no_mask(sz):
    'all zero mask'
    return torch.zeros(sz, sz)

def generate_lookback_mask(sz, look_back):
    mask1 = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    mask2 = torch.tril(torch.ones(sz, sz) * float('-inf'), diagonal=-look_back)
    return mask1 + mask2