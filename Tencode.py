import torch
import torch.nn as nn
# import pytorch_lightning as pt
import lightning as pt
from einops import rearrange
from lightning import Trainer
from torch.utils.data import DataLoader

from base_modle.base_modle import cov_encode, ATT_encode, EMBDim, res_modle
from base_modle.dataset import dastset


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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.002)

        return {"optimizer": optimizer,
                # "lr_scheduler": lt
                }


    def training_step(self, batch, batch_idx):
        img_tensor, tocken, padmask, masktocken=batch

        img,bh=self.forward(img_tensor,masktocken,bhmask=padmask)

        loss_img=nn.SmoothL1Loss()(img_tensor,img)

        # bh = rearrange(bh, 'b n s -> b s n ', )
        # bhloss=nn.CrossEntropyLoss()(bh,tocken.long())
        bhloss=0

        return (loss_img+bhloss)/2




if __name__=='__main__':

    modss=FORT_encode(ATTlays=5,bhlay=6,imglay=4,dim=512,heads=8,inner_dim=512,out_dim=48,pos_emb_drop=0.1,mlpdropout=0.5,attdropout=0.05)
    aaaa = dastset('映射.json', 'fix1.json', './i')

    from pytorch_lightning import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"lagegeFDbignet_1000")


    trainer = Trainer(accelerator='gpu',gradient_clip_val=0.1,logger=tensorboard,max_epochs=400
                      #, ckpt_path=r'C:\Users\autumn\Desktop\poject_all\Font_DL\lightning_logs\version_41\checkpoints\epoch=34-step=70000.ckpt'
                      )
    # trainer.save_checkpoint('test.pyt')
    trainer.fit(model=modss,train_dataloaders=DataLoader(dataset=aaaa,batch_size=6
                                                   #,num_workers=1
                                                   ))






    aaa=FORT_encode(5,5,5,512,8,512,32)
    aaa=aaa(torch.randn(4,3,256,256),torch.ones(4,12,dtype=torch.int32),bhmask=torch.ones(4,12))
    aaa










