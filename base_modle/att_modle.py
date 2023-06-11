import random

import torch

from torch import nn
import math
from functools import partial

from torch.nn.utils import weight_norm

import torch.nn.functional as F
from einops import rearrange, repeat


class attention(nn.Module):
    def __init__(self, dim, heads, inner_dim, dropout=0.0, att_type=None):
        super().__init__()
        if att_type is None:
            att_type = {}
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.dim_head = dim / heads
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.att_type = att_type

    def forward(self, k, v, q, attn_mask=None):

        q, k, v = self.to_q(q), self.to_k(v), self.to_v(k)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        with torch.backends.cuda.sdp_kernel(**self.att_type):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, )

        out = rearrange(x, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.Lin1 = nn.Linear(dim, dim)
        self.Lin1 = nn.Linear(dim, dim)

    def forward(self, x):
        out, gate = self.Lin1(x), self.Lin2(x)
        return out * gate.sigmoid()


class MLP_T1(nn.Module):
    def __init__(self, dim, jhhc='GELU', dropout=0.0, pox=4):
        super().__init__()
        self.Lin1 = nn.Linear(dim, pox * dim)

        if jhhc == 'GELU':
            self.hc = nn.GELU()
        elif jhhc == 'Swish':
            self.hc = swish
        elif jhhc == 'GLU':
            self.hc = GLU(dim=dim)
        else:
            raise Exception('不支持方法')

        self.Lin2 = nn.Linear(pox * dim, dim)
        self.drp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.Lin1(x)
        x = self.hc(x)
        x = self.Lin2(x)
        return self.drp(x)


class cross_att_lay(nn.Module):
    def __init__(self, dim, heads, inner_dim, mlpdropout=0.0, pox=4, attdropout=0.0, att_type=None, jhhc='GELU'):
        super().__init__()
        self.attention_bh = attention(dim, heads, inner_dim, attdropout, att_type)
        self.attention_img = attention(dim, heads, inner_dim, attdropout, att_type)

        self.Mlp_bh = MLP_T1(dim, jhhc, mlpdropout, pox)
        self.Mlp_img = MLP_T1(dim, jhhc, mlpdropout, pox)
        self.ln_bh1 = nn.LayerNorm(dim, eps=1e-12)
        self.ln_bh2 = nn.LayerNorm(dim, eps=1e-12)
        self.ln_img1 = nn.LayerNorm(dim, eps=1e-12)
        self.ln_img1 = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x_img, y_bh, bh_attention_mask=None, img_attention_mask=None):
        y_bhs = self.ln_bh1(self.attention_bh(x_img, x_img, y_bh, bh_attention_mask) + y_bh)
        y_bhs = self.ln_bh2(self.Mlp_bh(y_bhs) + y_bhs)

        x_img = self.ln_bh1(self.attention_bh(y_bh, y_bh, x_img, img_attention_mask) + x_img)
        x_img = self.ln_bh2(self.Mlp_bh(x_img) + x_img)

        return x_img, y_bhs,


class self_att_lay(nn.Module):
    def __init__(self, dim, heads, inner_dim, mlpdropout=0.0, pox=4, attdropout=0.0, att_type=None, jhhc='GELU'):
        super().__init__()
        self.attention_bh = attention(dim, heads, inner_dim, attdropout, att_type)

        self.Mlp_bh = MLP_T1(dim, jhhc, mlpdropout, pox)

        self.ln_bh1 = nn.LayerNorm(dim, eps=1e-12)
        self.ln_bh2 = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x, attention_mask=None):
        x = self.ln_bh1(self.attention_bh(x, x, x, attention_mask) + x)
        x = self.ln_bh2(self.Mlp_bh(x) + x)

        return x


class coross_attention(nn.Module):
    def __init__(self, cross_lays, dim, heads, inner_dim, mlpdropout=0.0, attdropout=0.0, pox=4, att_type=None,
                 jhhc='GELU'):
        super().__init__()
        self.lix = nn.ModuleList()
        for i in range(cross_lays):
            self.lix.append(cross_att_lay(dim, heads, inner_dim, mlpdropout, pox, attdropout, att_type, jhhc))

    def forward(self, x_img, y_bh, bh_attention_mask=None, img_attention_mask=None):
        for lay in self.lix:
            x_img, y_bh, = lay(x_img, y_bh, bh_attention_mask, img_attention_mask)
        return x_img, y_bh,


class self_attention(nn.Module):
    def __init__(self, lays, dim, heads, inner_dim, mlpdropout=0.0, attdropout=0.0, pox=4, att_type=None, jhhc='GELU'):
        super().__init__()
        self.lix = nn.ModuleList()
        for i in range(lays):
            self.lix.append(self_att_lay(dim, heads, inner_dim, mlpdropout, pox, attdropout, att_type, jhhc))

    def forward(self, x, attention_mask=None):
        for lay in self.lix:
            x = lay(x, attention_mask, )
        return x


if __name__ == '__main__':
    query = torch.rand(4, 8, 512, dtype=torch.float32, device="cuda")
    key = torch.ones(4, 4, 512, dtype=torch.float32, device="cuda")
    value = torch.rand(4, 4, 512, dtype=torch.float32, device="cuda")
    attn = torch.tensor([[True, True, True, False], [True, True, True, False], [True, True, True, False],
                         [False, False, False, False]]).cuda()
    ctc = attention(512, 8, 512).cuda()

    acs = ctc(key, value, query, attn)
    acs

    for i in range(100):
        asxd = random.randint(16, 1024)
        asxds = random.randint(16, 1024)
        asxdss = random.randint(16, 128)

        print(asxd, asxds, asxdss)
        query = torch.rand(asxdss, asxd, 512, dtype=torch.float32, device="cuda")
        key = torch.ones(asxdss, asxds, 512, dtype=torch.float32, device="cuda")
        value = torch.rand(asxdss, asxds, 512, dtype=torch.float32, device="cuda")
        acs = ctc(key, value, query, )
        print(acs.shape)

    img = torch.rand(4, 80, 512, dtype=torch.float32, device="cuda")
    bh = torch.ones(4, 42, 512, dtype=torch.float32, device="cuda")

    aaa = cross_att_lay(512, 8, 512).cuda()
    ccc = aaa(img, bh)
    ccc

    bbb = coross_attention(7, 512, 8, 512).cuda()
    dsds = bbb(img, bh)
    dsds

    bbbs = self_attention(7, 512, 8, 512).cuda()
    xdsdss = bbbs(img, )
    xdsdss
