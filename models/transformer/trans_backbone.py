import torch
import torch.nn as nn
from models.transformer.trans_encdec import Encoder, EncoderLayer, Decoder, DecoderLayer
from models.layers.embed import DataEmbedding
from models.layers.self_atten import FullAttention, AttentionLayer


class TransformerBackbone(nn.Module):
    def __init__(self, args):
        super(TransformerBackbone, self).__init__()
        self.args = args
        d_model = args.embed_dim
        n_heads = args.num_heads
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        self.enc_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
        self.dec_embedding = DataEmbedding(c_in=1, d_model=d_model)
        # Attention
        Attn = FullAttention
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
        if self.args.use_decoder:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        self_attention=AttentionLayer(Attn(mask_flag=True, attention_dropout=dropout, output_attention=args.output_attention),
                                    d_model, n_heads),
                        cross_attention=AttentionLayer(Attn(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                                    d_model, n_heads),
                        d_model=d_model, # type: ignore
                        d_ff=d_ff,
                        dropout=dropout,
                        activation=args.activation,
                    )
                    for l in range(args.num_dec_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        if self.args.cat_ret:
            x_enc = torch.concat([x_enc, y_enc], dim=-1)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, y_enc, atten_mask)
        if self.args.use_decoder:
            dec_inp = torch.zeros([y_enc.shape[0], self.args.pred_len, y_enc.shape[2]]).float().to(y_enc.device)
            dec_inp = torch.cat([y_enc[:, -self.args.dec_len_from_input:, :], dec_inp], dim=1).float().to(self.device)
            dec_out = self.dec_embedding(dec_inp)
            dec_out = self.decoder(dec_out, enc_out)
        else:
            dec_out = enc_out
        return dec_out  # [B, L, D]


