from base_modle.att_modle import self_attention, coross_attention
import torch

from torch import nn
import math
from functools import partial

from torch.nn.utils import weight_norm

import torch.nn.functional as F


class encode(nn.Module):
    def __init__(self, lays, dim, heads, inner_dim, out_dim, mlpdropout=0.0, attdropout=0.0, pox=4, att_type=None,
                 jhhc='GELU'):
        super().__init__()
        self.sfa_bh = self_attention(lays, dim, heads, inner_dim, mlpdropout, attdropout, pox, att_type, jhhc)
        self.sfa_img = self_attention(lays, dim, heads, inner_dim, mlpdropout, attdropout, pox, att_type, jhhc)
        self.croa = coross_attention(lays, dim, heads, inner_dim, mlpdropout, attdropout, pox, att_type, jhhc)

        self.bh_out = nn.Linear(dim, out_dim)

    def forward(self, x_img, y_bh, bh_attention_mask=None, img_attention_mask=None):
        e_img = self.sfa_img(x_img, img_attention_mask)
        e_bh = self.sfa_bh(y_bh, bh_attention_mask)

        img, bh = self.croa(e_img, e_bh, bh_attention_mask, img_attention_mask)
        return img, self.bh_out(bh)
