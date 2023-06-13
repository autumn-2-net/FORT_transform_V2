
import torch
from torch import nn
from base_modle.SWN import SwitchNorm2d

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, k_size=3 ):
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel,inchannel,kernel_size=(3,3),padding=1)
        # self.bn1 = SwitchNorm2d(inchannel)
        self.bn1 = nn.GroupNorm(4,inchannel)
        self.relu = nn.ReLU()
        # self.relu = GLU(1)
        self.conv2 = nn.Conv2d(inchannel,inchannel,kernel_size=(3,3),padding=1)
        # self.conv3 = nn.Conv2d(inchannel, inchannel , kernel_size=(3, 3), padding=1)
        # self.bn2 = SwitchNorm2d(inchannel)
        self.bn2 = nn.GroupNorm(4,inchannel)
        self.eca = eca_layer(inchannel, k_size)
        self.relus = nn.LeakyReLU()




    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)


        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        # out = self.conv3(out)

        out = self.eca(out)


        out += residual
        out = self.relu(out)



        return out