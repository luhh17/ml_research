import torch
import torch.nn as nn
from models.encoder import Encoder, TargetEncoderBlock, CausalEncoderBlock, EncoderLayer, CompressCausalEncoderLayer, \
    CompressTargetEncoderLayer
from models.layers.embed import DataEmbedding
from models.layers.self_atten import FullAttention, AttentionLayer, CausalAttention


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
        # self.dec_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
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
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             self_attention=AttentionLayer(Attn(mask_flag=True, attention_dropout=dropout, output_attention=args.output_attention),
        #                            d_model, n_heads),
        #             cross_attention=AttentionLayer(FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
        #                            d_model, n_heads),
        #             d_input=d_model,
        #             d_hidden=d_ff,
        #             dropout=dropout,
        #             activation=args.activation,
        #         )
        #         for l in range(args.num_dec_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(d_model)
        # )
        # self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        if self.args.cat_ret:
            x_enc = torch.concat([x_enc, y_enc], dim=-1)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, y_enc, atten_mask)
        # if self.args.use_decoder:
        #     dec_inp = build_decoder_input(self.args, x_enc[:, -self.args.dec_len_from_input:,], enc_out.shape[0], x_enc.shape[-1])
        #     B = dec_inp.shape[0]
        #     L = dec_inp.shape[1]
        #     attn_mask = TriangularCausalMask(B, L)
        #     dec_out = self.dec_embedding(dec_inp)
        #     dec_out = self.decoder(dec_out, enc_out, x_mask=attn_mask, cross_mask=None)
        # else:
        dec_out = enc_out
        if self.args.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]


class CausalTransformerBackbone(nn.Module):
    def __init__(self, args):
        super(CausalTransformerBackbone, self).__init__()
        self.args = args
        d_model = args.embed_dim
        n_heads = args.num_heads
        d_ff = args.hidden_dim
        dropout = args.dropout
        # Encoding
        self.enc_embedding = DataEmbedding(c_in=self.args.char_dim, d_model=d_model)
        self.y_embedding = DataEmbedding(c_in=1, d_model=d_model)
        # Attention
        self_attn = FullAttention
        causal_attn = CausalAttention
        # Encoder
        if self.args.only_one_target:
            self.encoder = nn.ModuleList([CausalEncoderBlock(args)] +
                                         [EncoderLayer(
                                             AttentionLayer(self_attn(mask_flag=False, attention_dropout=dropout,
                                                                      output_attention=args.output_attention),
                                                            d_model, n_heads), d_model, d_ff,
                                             dropout=dropout,
                                             activation=args.activation
                                         ) for l in range(args.num_enc_layers - 1)])
        else:
            self.encoder = nn.ModuleList([CausalEncoderBlock(args) for l in range(args.num_enc_layers)])

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        if self.args.cat_ret:
            x_enc = torch.concat([x_enc, y_enc], dim=-1)
        enc_out = self.enc_embedding(x_enc)
        y_enc = self.y_embedding(y_enc)
        for block in self.encoder:
            enc_out, y_enc = block(enc_out, y_enc, atten_mask)
        dec_out = enc_out
        return dec_out  # [B, L, D]


class CompressCausalTransformerBackbone(nn.Module):
    def __init__(self, args):
        super(CompressCausalTransformerBackbone, self).__init__()
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
                    CompressCausalEncoderLayer(args,
                        AttentionLayer(CausalAttention(args, mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                            d_model, n_heads),
                        AttentionLayer(Attn(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                                       d_model, n_heads), d_model, d_ff,
                        dropout=dropout,
                        activation=args.activation
                    )
                ] +
                [
                    EncoderLayer(AttentionLayer(Attn(mask_flag=False, attention_dropout=dropout,
                                            output_attention=args.output_attention),
                                      d_model, n_heads), d_model, d_ff,
                                             dropout=dropout,
                                             activation=args.activation
                                         ) for l in range(args.num_enc_layers-1)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            self.encoder = Encoder(
                [
                    CompressCausalEncoderLayer(
                        AttentionLayer(CausalAttention(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                            d_model, n_heads),
                        AttentionLayer(Attn(mask_flag=False, attention_dropout=dropout, output_attention=args.output_attention),
                                       d_model, n_heads), d_model, d_ff,
                        dropout=dropout,
                        activation=args.activation
                    ) for l in range(args.num_enc_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        enc_out = self.enc_embedding(x_enc)
        y_enc = self.y_embedding(y_enc)
        enc_out, attns = self.encoder(enc_out, y_enc, atten_mask)
        dec_out = enc_out
        if self.args.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]


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
        # Attention
        self_attn = FullAttention
        causal_attn = CausalAttention
        # Encoder


        if self.args.only_one_target:
            self.encoder = nn.ModuleList([TargetEncoderBlock(args)] +
                                         [EncoderLayer(
                                             AttentionLayer(self_attn(mask_flag=False, attention_dropout=dropout,
                                                                      output_attention=args.output_attention),
                                                            d_model, n_heads), d_model, d_ff,
                                             dropout=dropout,
                                             activation=args.activation
                                         ) for l in range(args.num_enc_layers - 1)])
        else:
            self.encoder = nn.ModuleList([TargetEncoderBlock(args) for l in range(args.num_enc_layers)])

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        if self.args.cat_ret:
            x_enc = torch.concat([x_enc, y_enc], dim=-1)
        enc_out = self.enc_embedding(x_enc)
        y_enc = self.y_embedding(y_enc)
        for block in self.encoder:
            enc_out, y_enc = block(enc_out, y_enc, atten_mask)
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
