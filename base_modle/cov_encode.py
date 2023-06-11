import torch
import torch.nn as nn
import pytorch_lightning as pyl

from base_modle.res_modle import res_modle


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

        # return a


# a =cov_encode()
# c = torch.FloatTensor(3,256,256)
# a =a(c)
# print(a)
