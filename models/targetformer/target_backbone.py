import torch
import torch.nn as nn
from models.targetformer.target_encdec import Encoder, TargetEncoderBlock, EncoderLayer, CompressTargetEncoderLayer
from models.layers.embed import DataEmbedding
from models.layers.self_atten import FullAttention, AttentionLayer, CausalAttention


class TargetTransformerBackbone(nn.Module):
    def __init__(self, args):
        super(TargetTransformerBackbone, self).__init__()
        self.args = args
        d_model = args.embed_dim
        n_heads = args.num_heads
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        self.enc_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
        self.y_embedding = DataEmbedding(c_in=1, d_model=d_model)
        # Encoder
        if self.args.only_one_target:
            self.encoder = nn.ModuleList([TargetEncoderBlock(args)] +
                                         [EncoderLayer(
                                             AttentionLayer(FullAttention(mask_flag=False, attention_dropout=dropout,
                                                                      output_attention=args.output_attention),
                                                            d_model, n_heads), d_model, d_ff,
                                             dropout=dropout,
                                             activation=args.activation, prenorm=self.args.prenorm
                                         ) for l in range(args.num_enc_layers - 1)])
        else:
            self.encoder = nn.ModuleList([TargetEncoderBlock(args) for l in range(args.num_enc_layers)])
        self.decoder = None

        # if self.args.use_decoder:
        #     self.decoder = nn.ModuleList([EncoderLayer(
        #         AttentionLayer(CausalAttention(attention_dropout=dropout), d_model, n_heads), d_model, d_ff,
        #         dropout=dropout,
        #         activation=args.activation
        #     ) for l in range(args.num_dec_layers)])
        

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        if self.args.cat_ret:
            x_enc = torch.concat([x_enc, y_enc], dim=-1)
        enc_out = self.enc_embedding(x_enc)
        y_enc = self.y_embedding(y_enc)
        for block in self.encoder:
            enc_out, y_enc = block(enc_out, y_enc, atten_mask)
        # if self.decoder is not None:
        #     for block in self.decoder:
        #         enc_out, _ = block(enc_out, None, atten_mask)
        dec_out = enc_out
        return dec_out  # [B, L, D]


class CompressTargetTransformerBackbone(nn.Module):
    def __init__(self, args):
        super(CompressTargetTransformerBackbone, self).__init__()
        self.args = args
        d_model = args.embed_dim
        n_heads = args.num_heads
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        self.enc_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
        self.y_embedding = DataEmbedding(c_in=1, d_model=d_model)
        Attn = FullAttention
        # Encoder
        if self.args.only_one_target:
            self.encoder = Encoder(
                [
                    CompressTargetEncoderLayer(args,
                                               AttentionLayer(
                                                   Attn(mask_flag=False, attention_dropout=dropout,
                                                        output_attention=args.output_attention),
                                                   d_model, n_heads), d_model, d_ff,
                                               dropout=dropout,
                                               activation=args.activation
                                               )
                ] +
                [
                    EncoderLayer(
                                             AttentionLayer(Attn(mask_flag=False, attention_dropout=dropout,
                                                                      output_attention=args.output_attention),
                                                            d_model, n_heads), d_model, d_ff,
                                             dropout=dropout,
                                             activation=args.activation
                                         ) for l in range(args.num_enc_layers - 1)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            self.encoder = Encoder(
                [
                    CompressTargetEncoderLayer(args,
                        AttentionLayer(
                            Attn(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                            d_model, n_heads), d_model, d_ff,
                        dropout=dropout,
                        activation=args.activation
                    ) for l in range(args.num_enc_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        if self.args.cat_ret:
            x_enc = torch.concat([x_enc, y_enc], dim=-1)
        enc_out = self.enc_embedding(x_enc)
        y_enc = self.y_embedding(y_enc)
        enc_out, target = self.encoder(enc_out, y_enc, atten_mask)
        dec_out = enc_out

        return dec_out  # [B, L, D]
