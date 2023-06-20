import random

import numpy
import torch
from einops import rearrange
from torch import nn

from  base_modle.base_modle import cov_encode
from  base_modle.dataset import dastset
from base_modle.st2_ds_h5 import st2_dataset
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


if __name__=='__main__':



    # fff=st2_dataset('V2_dataset_stage2.hdf5','st2_rcmap','st2_map','./i/','img_mapss')
    # for i in fff:
    #     i
    # pass
    # asfsd=nn.Sequential(
    #     nn.ConvTranspose2d(in_channels=512, out_channels=600, kernel_size=(15, 15), stride=2,
    #                        padding=0), GLU(1),
    #     nn.ConvTranspose2d(in_channels=300, out_channels=512, kernel_size=(15, 15), stride=2,
    #                        padding=0), GLU(1),
    #     nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(8, 8), stride=2,
    #                        padding=1), GLU(1),
    #     nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(8, 8), stride=2,
    #                        padding=2), GLU(1),
    #     nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(8, 8), stride=2,
    #                        padding=1),
    #     # nn.Sigmoid()
    #     nn.ReLU(),
    # )

    aaaaaa=torch.load('C:/Users/autumn/Downloads/新建文件夹 (19)/1model.ckpt')
    aaaaaa1 = torch.load('C:/Users/autumn/Downloads/vgg.pth')
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