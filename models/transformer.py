import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer
from models.decoder import Decoder, DecoderLayer
from models.layers.embed import DataEmbedding
from models.layers.self_atten import FullAttention, AttentionLayer, ProbAttention
from utils.data_organization import build_decoder_input
from models.layers.head import RegressionHead, PretrainHead
from models.backbone import TransformerBackbone, TargetTransformerBackbone, CompressTargetTransformerBackbone, CausalTransformerBackbone, CompressCausalTransformerBackbone


class Transformer(nn.Module):
    # transformer for prediction
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        d_model = args.embed_dim
        dropout = args.dropout
        # Encoding
        if self.args.atten == 'compress_target':
            self.backbone = CompressTargetTransformerBackbone(args)
        elif self.args.atten == 'target':
            self.backbone = TargetTransformerBackbone(args)
        elif self.args.atten == 'compress_causal':
            self.backbone = CompressCausalTransformerBackbone(args)
        elif self.args.atten == 'causal':
            self.backbone = CausalTransformerBackbone(args)
        elif self.args.atten == 'full':
            self.backbone = TransformerBackbone(args)
        self.head = RegressionHead(d_model=d_model, output_dim=args.pred_length, head_dropout=dropout, num_patch=args.window_length, flatten=True)

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        enc_out = self.backbone(x_enc, y_enc, atten_mask)
        out = self.head(enc_out)
        return out


class TransformerPretrain(nn.Module):
    def __init__(self, args):
        super(TransformerPretrain, self).__init__()
        self.args = args
        d_model = args.embed_dim
        dropout = args.dropout
        # Encoding
        self.backbone = TransformerBackbone(args)
        if self.args.mask_target == 'signal':
            self.head = PretrainHead(d_model=d_model, output_dim=args.char_dim, head_dropout=dropout)
        elif self.args.mask_target == 'ret':
            self.head = PretrainHead(d_model=d_model, output_dim=1, head_dropout=dropout)

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        enc_out = self.backbone(x_enc, y_enc, atten_mask)
        out = self.head(enc_out)
        return out


class Informer(nn.Module):
    def __init__(self, args):
        super(Informer, self).__init__()
        self.args = args
        d_model = args.embed_dim
        n_heads = args.num_heads
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        self.enc_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
        self.dec_embedding = DataEmbedding(c_in=1, d_model=d_model)
        # Attention
        Attn = ProbAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                                   d_model, n_heads), d_model, d_ff,
                    dropout=dropout,
                    activation=args.activation
                ) for l in range(args.num_enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention=AttentionLayer(Attn(mask_flag=True, attention_dropout=dropout, output_attention=args.output_attention),
                                   d_model, n_heads),
                    cross_attention=AttentionLayer(FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                                   d_model, n_heads),
                    d_input=d_model,
                    d_hidden=d_ff,
                    dropout=dropout,
                    activation=args.activation,
                )
                for l in range(args.num_dec_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_enc, y_enc=None, x_mark=None):

        enc_out = self.enc_embedding(x_enc, x_mark)
        enc_out, attns = self.encoder(enc_out)
        if self.args.use_decoder:
            dec_inp = build_decoder_input(self.args, y_enc, enc_out.shape[0], 1)
            dec_out = self.dec_embedding(dec_inp, x_mark)
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
            dec_out = self.projection(dec_out)
        else:
            dec_out = self.projection(enc_out)

        return dec_out[:, -self.args.pred_length:, :]  # [B, L, D]


class SerialIterModel(nn.Module):
    def __init__(self, args, base_model):
        super(SerialIterModel, self).__init__()
        self.args = args
        self.model = base_model

    def forward(self, x_enc, input_mask=None):
        # input x: N X T X C
        # transformer needs input :[B, L, D]
        # tcn needs input: N x C x L
        # output T x N
        if 'former' in self.args.model or 'tlinear' in self.args.model:
            dec_list = []
            for idx in range(self.args.max_steps):
                x_input = x_enc[:, idx: idx+self.args.window_length, :]
                dec_out = self.model(x_input)
                dec_out = dec_out[:, -self.args.pred_length:, :]
                dec_list.append(dec_out)
            res = torch.cat(dec_list, dim=1)
            res = res.squeeze(-1)
            return res
        elif self.args.model == 'tcn':
            x_enc = x_enc.permute(0, 2, 1)
            dec_list = []
            for idx in range(self.args.max_steps):
                x_input = x_enc[:, :, idx: idx + self.args.window_length]
                dec_out = self.model(x_input)
                dec_out = dec_out[:, -self.args.pred_length:, :]
                dec_list.append(dec_out)
            res = torch.cat(dec_list, dim=1).squeeze(-1)
            return res
