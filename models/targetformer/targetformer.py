import torch
import torch.nn as nn
from models.layers.head import RegressionHead
from models.targetformer.target_backbone import TargetTransformerBackbone


class Targetformer(nn.Module):
    # transformer for prediction
    def __init__(self, args, all_steps=False):
        super(Targetformer, self).__init__()
        self.args = args
        d_model = args['embed_dim']
        dropout = args['dropout']
        # Encoding
        self.backbone = TargetTransformerBackbone(args)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.head = RegressionHead(d_model=d_model, output_dim=args['pred_length'], head_dropout=dropout, num_patch=args['window_length'], flatten=all_steps)

    def forward(self, x_enc, y_enc=None, atten_mask=None):
        enc_out = self.backbone(x_enc, y_enc, atten_mask)
        enc_out = self.batchnorm(enc_out.transpose(1, 2)).transpose(1, 2)
        out = self.head(enc_out)
        return out