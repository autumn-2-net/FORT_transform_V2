from base_modle.att_modle import self_attention, coross_attention
from base_modle.positemb import RelPositionalEncoding
import torch

from torch import nn
import math
from functools import partial

from torch.nn.utils import weight_norm

import torch.nn.functional as F


class ATT_encode(nn.Module):
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

class res_block(nn.Module):
    def __init__(self,covsiz,chanal,stride,padding=0):

        super().__init__()
        self.cov =nn.Conv2d(in_channels=chanal,out_channels=chanal,kernel_size=covsiz,stride=stride,padding=padding)
        self.relu =nn.SELU()

    def swish(self,x):
        a =nn.Sigmoid()
        return x*a(x)

    def forward(self,x):
        a =x
        b =self.cov(x)
        # b=self.relu(b)
        b = self.swish(b)
        return a+b

class res_modle(nn.Module):
    def __init__(self,n_layer,chanal,):
        super().__init__() #cov4 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(3, 3),padding=1)

        self.res_net =nn.Sequential(*[res_block(chanal=chanal,covsiz=(3,3),stride=1,padding=1) for res in range(n_layer)]) #        self.blocks = nn.Sequential(*[GPT_Block(config = config) for _ in range(config.n_layer)])
    def forward(self, x):
        return self.res_net(x)
class cov_encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(8, 8), stride=2,padding=1), nn.ELU(),
                                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8, 8), stride=2,padding=1), nn.ELU(),
                                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(8, 8), stride=2,padding=1), nn.ELU(),
                                  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(15, 15), stride=2,padding=1), nn.ELU(),
                                  )
        self.cov2 =res_modle(n_layer=4,chanal=512)

    def forward(self, x):
        a =self.cov1(x)
        return self.cov2(a)
    
class EMBDim(nn.Module):
    def __init__(self,dim,postlen=65,embt='REL',):
        super().__init__()
        self.type_emb=nn.Embedding(2,dim)
        self.posTYPE=embt
        self.position = nn.Embedding(num_embeddings=postlen, embedding_dim=dim)

    def get_post(self,x,mask=None):
        if self.posTYPE == 'REL':

