import random

import torch
from einops import rearrange

from  base_modle.base_modle import cov_encode
from  base_modle.dataset import dastset

if __name__=='__main__':
    aaaa=dastset('映射.json','fix1.json','./i')
    for i in range(1000):
        asa=random.randint(0,44230)
        e=aaaa[asa]
        pass

    a=torch.randn(4,512,8,8)

    out = rearrange(a, 'b c h w -> b (h w) c')
    pass



    aaa=cov_encode()(torch.randn(20,3,256,256))
    aaa