import torch
import torch.nn as nn
# import pytorch_lightning as pt
import lightning as pt
from einops import rearrange
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from base_modle.base_modle import cov_encode, ATT_encode, EMBDim, res_modle,PATT_encode,Gres_modle
from base_modle.dataset import dastset,Fdastset
from matplotlib import pyplot as plt

from base_modle.scheduler import WarmupLR
from base_modle.ECA_net import ECABasicBlock

from base_modle.st2_ds_h5 import st2_dataset

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class PFORT_DECODE(pt.LightningModule):
    def __init__(self, dim, eaclay):
        super().__init__()
        self.eacnet=nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])
        self.eacnet2 = nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])

        # self.eacnet = Gres_modle(eaclay,dim)
        # self.eacnet = res_modle(eaclay, dim)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim*2, out_channels=512, kernel_size=(15, 15), stride=2,
                               padding=0), GLU(1),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(8, 8), stride=2,
                               padding=1), GLU(1),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(8, 8), stride=2,
                               padding=2), GLU(1),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(8, 8), stride=2,
                               padding=1),
            # nn.Sigmoid()
            nn.Softplus(),
            )
        # self.decode = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=dim, out_channels=256, kernel_size=(15, 15), stride=2,
        #                        padding=0), nn.ELU(),
        #     nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(8, 8), stride=2,
        #                        padding=1), nn.ELU(),
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(8, 8), stride=2,
        #                        padding=2), nn.ELU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(8, 8), stride=2,
        #                        padding=1),
        #     # nn.Sigmoid()
        #     nn.Softplus(),
        # )
        self.grad_norm = 0
        self.lrc = 0.0001




    def forward(self,img_feature1,img_feature2):
        img_feature1=self.eacnet(img_feature1)
        img_feature2 = self.eacnet2(img_feature2)
        # img_feature1, _= img_feature1.chunk(2, dim=1)
        # _, img_feature2 = img_feature2.chunk(2, dim=1)

        feature=torch.cat((img_feature1,img_feature2),dim=1)
        img=self.decode(feature)

        return img

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        # lt = {
        #     "scheduler": WarmupLR(optimizer, 5000, 1e-4),  # 调度器
        #     "interval": 'step',  # 调度的单位，epoch或step
        #
        #     "reduce_on_plateau": False,  # ReduceLROnPlateau
        #     "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
        #     "strict": False  # 如果没有monitor，是否中断训练
        # }

        return {"optimizer": optimizer,
                # "lr_scheduler": lt
                }


    def training_step(self, batch, batch_idx):
        img_f1, img_f2, t_img, img1,img2=batch

        img=self.forward(img_f1,img_f2)

        # loss_img=nn.SmoothL1Loss()(img_tensor,img)
        loss_img = nn.L1Loss()(t_img, img)


        # bhloss=0

        if batch_idx%50==0:
            self.wwww(batch_idx,loss_img, t_img, img1,img2,img)


        return loss_img
    # def on_after_backward(self):
    #     self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(),  1e9)
    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def wwww(self,batch_idx,loss1, t_img, img1,img2,oimg):


        step=self.global_step
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)


        writer.add_scalar('train/loss1', loss1, step)
        # writer.add_scalar('train/loss2', loss2, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.add_scalar('train/lr', self.lrc, step)
        if batch_idx % 200 == 0:
            writer.add_images('train_img/out',oimg.float(), step)
            writer.add_images('train_img/GTimg', t_img.float(), step)
            writer.add_images('train_img/INimg1', img1.float(), step)
            writer.add_images('train_img/INimg2', img2.float(), step)
        # if batch_idx % 100 == 0:
        #     GT,pre,mask = self.mcpx(bh,tocken,masktocken)
        #     writer.add_figure('M/GT', GT, step)
        #     writer.add_figure('M/pre', pre, step)
        #     writer.add_figure('M/mask', mask, step)
        writer.flush()





if __name__=='__main__':
    writer = SummaryWriter("./st2_log/", )
    modss=PFORT_DECODE(dim=512,eaclay=5)
    # aaaa = dastset('映射.json', 'fix1.json', './i')
    aaaa = st2_dataset('V2_dataset_stage2.hdf5','st2_rcmap','st2_map','./i/','img_mapss')
    from pytorch_lightning import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"FORT_modle_ST2")
    checkpoint_callback = ModelCheckpoint(

        # monitor = 'val/loss',

        dirpath='./mdscpss',

        filename='sample-mnist-epoch{epoch:02d}-{epoch}-{step}',

        auto_insert_metric_name=False#, every_n_epochs=20
        , save_top_k=-1,every_n_train_steps=30000

    )

    trainer = Trainer(accelerator='gpu',logger=tensorboard,max_epochs=400,callbacks=[checkpoint_callback],#precision='bf16'
                      #, ckpt_path=r'C:\Users\autumn\Desktop\poject_all\Font_DL\lightning_logs\version_41\checkpoints\epoch=34-step=70000.ckpt'
                      )
    # trainer.save_checkpoint('test.pyt')
    trainer.fit(model=modss,train_dataloaders=DataLoader(dataset=aaaa,batch_size=4,shuffle=True
                                                   ,num_workers=4,prefetch_factor =16,pin_memory=True,
                                                   ))





