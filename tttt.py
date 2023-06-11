import torch
from einops import rearrange

from  base_modle.base_modle import cov_encode

if __name__=='__main__':

    a=torch.randn(4,512,8,8)

    out = rearrange(a, 'b c h w -> b (h w) c')




    aaa=cov_encode()(torch.randn(20,3,256,256))
    aaa