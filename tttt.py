import random

import numpy
import torch
from einops import rearrange

from  base_modle.base_modle import cov_encode
from  base_modle.dataset import dastset
from base_modle.st2_ds_h5 import st2_dataset

if __name__=='__main__':

    fff=st2_dataset('V2_dataset_stage2.hdf5','st2_rcmap','st2_map','./i/','img_mapss')
    for i in fff:
        i
    pass


    #
    # aaaa=dastset('映射.json','fix1.json','./i')
    # for i in range(1000):
    #     asa=random.randint(0,44230)
    #     e=aaaa[asa]
    #     pass
    #
    # a=torch.randn(4,512,8,8)
    #
    # out = rearrange(a, 'b c h w -> b (h w) c')
    # pass
    #
    #
    #
    # aaa=cov_encode()(torch.randn(20,3,256,256))
    # aaa