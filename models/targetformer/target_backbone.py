import torch
import torch.nn as nn
from models.targetformer.target_encdec import Encoder, TargetEncoderBlock, EncoderLayer, CompressTargetEncoderLayer
from models.layers.embed import DataEmbedding
from models.layers.self_atten import FullAttention, AttentionLayer, CausalAttention


class TargetTransformerBackbone(nn.Module):
    def __init__(self, args):
        super(TargetTransformerBackbone, self).__init__()
        self.args = args
        d_model = args['embed_dim']
        n_heads = args['num_heads']
        d_ff = args['hidden_dim']
        dropout = args['dropout']
        # Encoding
        self.enc_embedding = DataEmbedding(c_in=self.args['char_dim'], d_model=d_model)
        self.y_embedding = DataEmbedding(c_in=1, d_model=d_model)
        # Encoder
        
        self.encoder = nn.ModuleList([TargetEncoderBlock(args)] +
                                        [EncoderLayer(
                                            AttentionLayer(FullAttention(mask_flag=False, attention_dropout=dropout,
                                                                    output_attention=False),
                                                        d_model, n_heads), d_model, d_ff,
                                            dropout=dropout,
                                            activation=args['activation'], prenorm=True
                                        ) for l in range(args['num_enc_layers'] - 1)])
        
        

        

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        enc_out = self.enc_embedding(x_enc)
        y_enc = self.y_embedding(y_enc)
        for block in self.encoder:
            enc_out, y_enc = block(enc_out, y_enc, atten_mask)
        dec_out = enc_out
        return dec_out  # [B, L, D]
