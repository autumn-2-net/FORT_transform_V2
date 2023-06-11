import torch
import torch.nn as nn

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