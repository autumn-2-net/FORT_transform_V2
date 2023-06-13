import torch
import torch.nn as nn
# import pytorch_lightning as pt
import lightning as pt
from einops import rearrange
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from base_modle.base_modle import cov_encode, ATT_encode, EMBDim, res_modle,PATT_encode
from base_modle.dataset import dastset,Fdastset
from matplotlib import pyplot as plt

from base_modle.scheduler import WarmupLR


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
                 jhhc='Swish',rea_lays=4):
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
        self.grad_norm = 0
        self.lrc = 0.0002




    def forward(self,x_img,y_bh,imgmask=None,bhmask=None):
        img_feature= rearrange(self.resc(self.in_cov(x_img)), 'b c h w -> b (h w) c')
        img_feature=self.EMA(img_feature,0,imgmask)

        bh_feature=self.EMA(y_bh,1,bhmask)

        img_feature,bh=self.ATT(img_feature,bh_feature, bh_attention_mask=bhmask, img_attention_mask=imgmask)
        img_feature=rearrange(img_feature, 'b (h w) c -> b c h w',h=8)

        img =self.decode(img_feature)
        return img,bh

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        lt = {
            "scheduler": WarmupLR(optimizer, 25000, 5e-5),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lt
                }


    def training_step(self, batch, batch_idx):
        img_tensor, tocken, padmask, masktocken=batch

        img,bh=self.forward(img_tensor,masktocken,bhmask=padmask)

        # loss_img=nn.SmoothL1Loss()(img_tensor,img)
        loss_img = nn.L1Loss()(img_tensor, img)

        bhs = rearrange(bh, 'b n s -> b s n ', )
        bhloss=nn.CrossEntropyLoss(ignore_index=0)(bhs,tocken.long())
        # bhloss=0

        if batch_idx%50==0:
            self.wwww(batch_idx,img,img_tensor,loss_img,bhloss,bh,tocken,masktocken)


        return (loss_img+bhloss)/2
    # def on_after_backward(self):
    #     self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(),  1e9)
    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def wwww(self,batch_idx,img,img2,loss1,loss2,bh,tocken,masktocken):


        step=self.global_step
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)


        writer.add_scalar('train/loss1', loss1, step)
        writer.add_scalar('train/loss2', loss2, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.add_scalar('train/lr', self.lrc, step)
        if batch_idx % 200 == 0:
            writer.add_images('train/img',img.float(), step)
            writer.add_images('train/gtimg', img2.float(), step)

        if batch_idx % 100 == 0:
            GT,pre,mask = self.mcpx(bh,tocken,masktocken)
            writer.add_figure('M/GT', GT, step)
            writer.add_figure('M/pre', pre, step)
            writer.add_figure('M/mask', mask, step)
        writer.flush()

    def mcpx(self,bh:torch.Tensor,tocken,masktocken):
        # import random

        # from matplotlib import cm
        # from matplotlib import axes
        # from matplotlib.font_manager import FontProperties
        # font = FontProperties(fname='/Library/Fonts/Songti.ttc')
        aaa=nn.Softmax()(bh[0])

        cxcx=aaa.detach().cpu().numpy()

        tk =tocken[0].detach().cpu().numpy()
        ddd=torch.zeros(aaa.size())
        for i,idk in enumerate(tk):
            ddd[i][idk]=1
        ddd=ddd.cpu().numpy()

        tk2 = masktocken[0].detach().cpu().numpy()
        ddd2 = torch.zeros(aaa.size())
        for i, idk in enumerate(tk2):
            ddd2[i][idk] = 1
        ddd2 = ddd2.cpu().numpy()









            # 定义热图的横纵坐标

            # 准备数据阶段，利用random生成二维数据（5*5）

            # 作图阶段
        fig = plt.figure()
            # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(ddd, )
        # 增加右侧的颜色刻度条
        plt.colorbar(im2)
            # 定义横纵坐标的刻度
            # ax.set_yticks(range(len(yLabel)))
            # ax.set_yticklabels(yLabel, fontproperties=font)
            # ax.set_xticks(range(len(xLabel)))
            # ax.set_xticklabels(xLabel)
            # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(cxcx, )
            # 增加右侧的颜色刻度条
        plt.colorbar(im)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        im3 = ax3.imshow(ddd2, )
        plt.colorbar(im3)

        return fig,fig2,fig3
            # 增加标题
            # plt.title("This is a title", fontproperties=font)
            # show



class PFORT_encode(pt.LightningModule):
    def __init__(self, ATTlays, bhlay, imglay, dim, heads, inner_dim, out_dim, max_tochen_len=48, postlen=65, embt='REL',
                 pos_emb_drop=0.0, mlpdropout=0.0, attdropout=0.0, pox=4, att_type=None,
                 jhhc='Swish',rea_lays=4):
        super().__init__()
        self.ATT = PATT_encode(ATTlays, bhlay, imglay, dim, heads, inner_dim, out_dim, mlpdropout, attdropout, pox,
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
        self.grad_norm = 0
        self.lrc = 0.0002




    def forward(self,x_img,y_bh,imgmask=None,bhmask=None):
        img_feature= rearrange(self.resc(self.in_cov(x_img)), 'b c h w -> b (h w) c')
        img_feature=self.EMA(img_feature,0,imgmask)

        bh_feature=self.EMA(y_bh,1,bhmask)

        img_feature,bh=self.ATT(img_feature,bh_feature, bh_attention_mask=bhmask, img_attention_mask=imgmask)
        img_feature=rearrange(img_feature, 'b (h w) c -> b c h w',h=8)

        img =self.decode(img_feature)
        return img,bh

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        lt = {
            "scheduler": WarmupLR(optimizer, 25000, 5e-5),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lt
                }


    def training_step(self, batch, batch_idx):
        img_tensor, tocken, padmask, masktocken=batch

        img,bh=self.forward(img_tensor,masktocken,bhmask=padmask)

        # loss_img=nn.SmoothL1Loss()(img_tensor,img)
        loss_img = nn.L1Loss()(img_tensor, img)

        bhs = rearrange(bh, 'b n s -> b s n ', )
        bhloss=nn.CrossEntropyLoss(ignore_index=0)(bhs,tocken.long())
        # bhloss=0

        if batch_idx%50==0:
            self.wwww(batch_idx,img,img_tensor,loss_img,bhloss,bh,tocken,masktocken)


        return (loss_img+bhloss)/2
    # def on_after_backward(self):
    #     self.grad_norm = nn.utils.clip_grad_norm_(self.parameters(),  1e9)
    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def wwww(self,batch_idx,img,img2,loss1,loss2,bh,tocken,masktocken):


        step=self.global_step
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)


        writer.add_scalar('train/loss1', loss1, step)
        writer.add_scalar('train/loss2', loss2, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.add_scalar('train/lr', self.lrc, step)
        if batch_idx % 200 == 0:
            writer.add_images('train/img',img.float(), step)
            writer.add_images('train/gtimg', img2.float(), step)

        if batch_idx % 100 == 0:
            GT,pre,mask = self.mcpx(bh,tocken,masktocken)
            writer.add_figure('M/GT', GT, step)
            writer.add_figure('M/pre', pre, step)
            writer.add_figure('M/mask', mask, step)
        writer.flush()

    def mcpx(self,bh:torch.Tensor,tocken,masktocken):
        # import random

        # from matplotlib import cm
        # from matplotlib import axes
        # from matplotlib.font_manager import FontProperties
        # font = FontProperties(fname='/Library/Fonts/Songti.ttc')
        aaa=nn.Softmax()(bh[0])

        cxcx=aaa.detach().cpu().numpy()

        tk =tocken[0].detach().cpu().numpy()
        ddd=torch.zeros(aaa.size())
        for i,idk in enumerate(tk):
            ddd[i][idk]=1
        ddd=ddd.cpu().numpy()

        tk2 = masktocken[0].detach().cpu().numpy()
        ddd2 = torch.zeros(aaa.size())
        for i, idk in enumerate(tk2):
            ddd2[i][idk] = 1
        ddd2 = ddd2.cpu().numpy()









            # 定义热图的横纵坐标

            # 准备数据阶段，利用random生成二维数据（5*5）

            # 作图阶段
        fig = plt.figure()
            # 定义画布为1*1个划分，并在第1个位置上进行作图
        ax = fig.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(ddd, )
        # 增加右侧的颜色刻度条
        plt.colorbar(im2)
            # 定义横纵坐标的刻度
            # ax.set_yticks(range(len(yLabel)))
            # ax.set_yticklabels(yLabel, fontproperties=font)
            # ax.set_xticks(range(len(xLabel)))
            # ax.set_xticklabels(xLabel)
            # 作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(cxcx, )
            # 增加右侧的颜色刻度条
        plt.colorbar(im)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        im3 = ax3.imshow(ddd2, )
        plt.colorbar(im3)

        return fig,fig2,fig3
            # 增加标题
            # plt.title("This is a title", fontproperties=font)
            # show'

    def encode(self,x_img,y_bh,imgmask=None,bhmask=None):
        img_feature= rearrange(self.resc(self.in_cov(x_img)), 'b c h w -> b (h w) c')
        img_feature=self.EMA(img_feature,0,imgmask)

        bh_feature=self.EMA(y_bh,1,bhmask)

        img_feature,bh=self.ATT(img_feature,bh_feature, bh_attention_mask=bhmask, img_attention_mask=imgmask)
        img_feature = rearrange(img_feature, 'b (h w) c -> b c h w', h=8)

        return img_feature







if __name__=='__main__':
    writer = SummaryWriter("./mdsr_1000s/", )
    modss=PFORT_encode(ATTlays=5,bhlay=9,imglay=5,dim=512,heads=8,inner_dim=512,out_dim=48,pos_emb_drop=0.1,mlpdropout=0.05,attdropout=0.05)
    # aaaa = dastset('映射.json', 'fix1.json', './i')
    aaaa = dastset('映射.json', 'fix1.json', './i')
    from pytorch_lightning import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"FORT_modle")
    checkpoint_callback = ModelCheckpoint(

        # monitor = 'val/loss',

        dirpath='./mdscp',

        filename='sample-mnist-epoch{epoch:02d}-{epoch}-{step}',

        auto_insert_metric_name=False#, every_n_epochs=20
        , save_top_k=-1,every_n_train_steps=15000

    )

    trainer = Trainer(accelerator='gpu',logger=tensorboard,max_epochs=400,callbacks=[checkpoint_callback],#precision='bf16'
                      #, ckpt_path=r'C:\Users\autumn\Desktop\poject_all\Font_DL\lightning_logs\version_41\checkpoints\epoch=34-step=70000.ckpt'
                      )
    # trainer.save_checkpoint('test.pyt')
    trainer.fit(model=modss,train_dataloaders=DataLoader(dataset=aaaa,batch_size=4,shuffle=True
                                                   ,num_workers=4,prefetch_factor =16,pin_memory=True,
                                                   ))






    aaa=FORT_encode(5,5,5,512,8,512,32)
    aaa=aaa(torch.randn(4,3,256,256),torch.ones(4,12,dtype=torch.int32),bhmask=torch.ones(4,12))
    aaa










