import torch
import torch.nn as nn
from models.layers.embed import DataEmbedding
from models.autoformer.auto_corr import AutoCorrelation, AutoCorrelationLayer
from models.autoformer.auto_encdec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class AutoFormer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, args):
        super(AutoFormer, self).__init__()
        self.args = args
        self.seq_len = args.window_length
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        # Decomp
        kernel_size = args.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(args.char_dim, args.embed_dim, args.temp_embed, args.embed_type,
                                           args.dropout)
        self.dec_embedding = DataEmbedding(args.char_dim, args.embed_dim, args.temp_embed, args.embed_type,
                                           args.dropout)


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                        ),
                        args.embed_dim, args.num_heads),
                    args.embed_dim,
                    args.hidden_dim,
                    moving_avg=args.moving_avg,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.num_enc_layers)
            ],
            norm_layer=my_Layernorm(args.embed_dim)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, args.factor, attention_dropout=args.dropout,
                                        output_attention=False),
                        args.embed_dim, args.num_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, args.factor, attention_dropout=args.dropout,
                                        output_attention=False),
                        args.embed_dim, args.num_heads),
                    args.embed_dim,
                    1,
                    args.hidden_dim,
                    moving_avg=args.moving_avg,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.num_dec_layers)
            ],
            norm_layer=my_Layernorm(args.embed_dim),
            projection=nn.Linear(args.embed_dim, 1, bias=True)
        )
        self.projection = nn.Linear(args.embed_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, self.args.char_dim], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        if not self.args.use_decoder:
            dec_out = self.projection(enc_out)
            return dec_out[:, -self.pred_len:, :]
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final

        # dec_out = trend_part + seasonal_part
        dec_out = seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
