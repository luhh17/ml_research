import torch


class TriangularCausalMask():
    '''
    input: Batch Size X Seq_len
    return: Batch_Size X 1 X Seq_len X Seq_len (被广播到Head_num维度）
    生成左下三角阵，左下及对角线全为1,后续生成mask使得左下及对角线全为nan
    '''
    def __init__(self, B, L, diagonal=1):
        device = torch.device('cuda')
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=diagonal).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    '''
       input: Batch Size, Head_num, Seq_len
       return: Batch_Size X 1 X Seq_len X Seq_len
       生成上三角阵，左下及对角线全为0
       '''

    def __init__(self, B, H, L, index, scores):
        device = torch.device('cuda')
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def generate_lookback_mask(sz, look_back, device):
    with torch.no_grad():
        mask1 = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1).to(device)
        mask2 = torch.tril(torch.ones(sz, sz, dtype=torch.bool), diagonal=-look_back).to(device)
        return mask1 + mask2


