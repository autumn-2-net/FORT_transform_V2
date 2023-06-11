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
        self.dim_head=dim/heads
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.att_type = att_type



    def forward(self, k,v,q,attn_mask=None):

        q, k, v = self.to_q(q), self.to_k(v), self.to_v(k)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        if attn_mask is not None:
            attn_mask=attn_mask.unsqueeze(1).unsqueeze(1)

        with torch.backends.cuda.sdp_kernel(**self.att_type):
            x =F.scaled_dot_product_attention(q,k,v,attn_mask= attn_mask, )

        out = rearrange(x, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)





if __name__=='__main__':
    query = torch.rand(4,  8, 512, dtype=torch.float32, device="cuda")
    key = torch.ones(4,  4, 512, dtype=torch.float32, device="cuda")
    value = torch.rand(4,  4, 512, dtype=torch.float32, device="cuda")
    attn = torch.tensor([[True, True, True, False], [True, True, True, False], [True, True, True, False],
                         [False, False, False, False]]).cuda()
    ctc=attention(512,8,512).cuda()

    acs=ctc(key,value,query,attn)
    acs
    for i in range(100):
        asxd=random.randint(16,1024)
        asxds = random.randint(16, 1024)
        asxdss = random.randint(16, 128)

        print(asxd,asxds,asxdss)
        query = torch.rand(asxdss, asxd, 512, dtype=torch.float32, device="cuda")
        key = torch.ones(asxdss, asxds, 512, dtype=torch.float32, device="cuda")
        value = torch.rand(asxdss, asxds, 512, dtype=torch.float32, device="cuda")
        acs = ctc(key, value, query, )
        print(acs.shape)






