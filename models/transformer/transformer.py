import torch
import torch.nn as nn
from models.layers.head import RegressionHead
from models.transformer.trans_backbone import TransformerBackbone


class Transformer(nn.Module):
    # transformer for prediction
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        d_model = args.embed_dim
        dropout = args.dropout
        # Encoding
        self.backbone = TransformerBackbone(args)
        self.head = RegressionHead(d_model=d_model, output_dim=args.pred_length, head_dropout=dropout, num_patch=args.window_length, flatten=False)

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        enc_out = self.backbone(x_enc, y_enc, atten_mask)
        out = self.head(enc_out)
        return out