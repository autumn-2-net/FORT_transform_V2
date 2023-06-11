import torch

from  base_modle.base_modle import cov_encode

if __name__=='__main__':
    aaa=cov_encode()(torch.randn(20,3,256,256))
    aaa