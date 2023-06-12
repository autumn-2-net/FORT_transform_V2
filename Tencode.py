import torch
import torch.nn as nn
import pytorch_lightning as pt
from einops import rearrange

from base_modle.base_modle import cov_encode, ATT_encode, EMBDim, res_modle



class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class FORT_encode(pt.LightningModule):
    def __init__(self, ATTlays, bhlay, imglay, dim, heads, inner_dim, out_dim, max_tochen_len=48, postlen=65, embt='REL',
                 pos_emb_drop=0.0, mlpdropout=0.0, attdropout=0.0, pox=4, att_type=None,
                 jhhc='GELU',rea_lays=4):
        super().__init__()
        self.ATT = ATT_encode(ATTlays, bhlay, imglay, dim, heads, inner_dim, out_dim, mlpdropout, attdropout, pox,
                              att_type,
                              jhhc)

        self.EMA = EMBDim(dim, max_tochen_len=max_tochen_len, postlen=postlen, embt=embt, drop=pos_emb_drop)
        self.resc=res_modle(rea_lays,512)
        self.in_cov=cov_encode()

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=512, kernel_size=(15, 15), stride=2,
                               padding=0), GLU(1),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(8, 8), stride=2,
                               padding=1), GLU(1),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(8, 8), stride=2,
                               padding=2), GLU(1),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(8, 8), stride=2,
                               padding=1),
            # nn.Sigmoid()
            nn.ReLU(),
            )




    def forward(self,x_img,y_bh,imgmask=None,bhmask=None):
        img_feature= rearrange(self.resc(self.in_cov(x_img)), 'b c h w -> b (h w) c')
        img_feature=self.EMA(img_feature,0,imgmask)

        bh_feature=self.EMA(y_bh,1,bhmask)

        img_feature,bh=self.ATT(img_feature,bh_feature, bh_attention_mask=bhmask, img_attention_mask=imgmask)
        img_feature=rearrange(img_feature, 'b (h w) c -> b c h w',h=8)

        img =self.decode(img_feature)
        return img,bh






if __name__=='__main__':






    aaa=FORT_encode(5,5,5,512,8,512,32)
    aaa=aaa(torch.randn(4,3,256,256),torch.ones(4,12,dtype=torch.int32),bhmask=torch.ones(4,12))
    aaa










