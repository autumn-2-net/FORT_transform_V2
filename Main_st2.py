import os

from base_modle.att_modle import self_attention

os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

from base_modle.Wganu import calculate_gradient_penalty


import numpy as np
import torch
import torch.nn as nn
# import pytorch_lightning as pt
import lightning as pt
from einops import rearrange
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from base_modle.SWN import SwitchNorm2d
from base_modle.base_modle import cov_encode, ATT_encode, EMBDim, res_modle,PATT_encode,Gres_modle
from base_modle.st2_base import EMBDim
from base_modle.dataset import dastset,Fdastset
from matplotlib import pyplot as plt

from base_modle.scheduler import WarmupLR, V3LSGDRLR
from base_modle.ECA_net import ECABasicBlock

from base_modle.st2_ds_h5 import st2_dataset

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
class RGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.rrr=nn.Softplus()

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return self.rrr(out) * gate.sigmoid()


class PFORT_DECODE(pt.LightningModule):
    def __init__(self, dim, eaclay):
        super().__init__()
        self.eacnet=nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])
        # self.eacnet2 = nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])

        # self.eacnet = Gres_modle(eaclay,dim)
        # self.eacnet = res_modle(eaclay, dim)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=666, kernel_size=(5, 5), stride=2,
                               padding=1), GLU(1), SwitchNorm2d(333),
            nn.ConvTranspose2d(in_channels=333, out_channels=400, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(200),
            nn.ConvTranspose2d(in_channels=200, out_channels=300, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(150),
            nn.ConvTranspose2d(in_channels=150, out_channels=200, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(100),
            nn.ConvTranspose2d(in_channels=100, out_channels=3, kernel_size=(6, 6), stride=2,
                               padding=3),
            nn.Softplus()
            # nn.Softmax(),
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
        # ax5 = img_feature1.detach().cpu().numpy()
        # ax6 = img_feature2.detach().cpu().numpy()


        img_feature1=self.eacnet(img_feature1)
        img_feature2 = self.eacnet(img_feature2)
        # ax3 = img_feature1.detach().cpu().numpy()
        # ax4 = img_feature2.detach().cpu().numpy()

        img_feature1, _= img_feature1.chunk(2, dim=1)
        _, img_feature2 = img_feature2.chunk(2, dim=1)

        feature=torch.cat((img_feature1,img_feature2),dim=1)
        img=self.decode(feature)
        # cp1=img[1]
        # cp2 = img[0]
        # cpp=img.detach().cpu().numpy()
        # ax1=img_feature1.detach().cpu().numpy()
        # ax2=img_feature2.detach().cpu().numpy()
        pass

        return img

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        lt = {
            "scheduler": V3LSGDRLR(optimizer,),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lt
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

class Dx(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.cov1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=2, padding=2),
                                  GLU(1), SwitchNorm2d(32),
                                  nn.Conv2d(in_channels=32, out_channels=100, kernel_size=(5, 5), stride=2, padding=2),
                                  GLU(1), SwitchNorm2d(50),
                                  nn.Conv2d(in_channels=50, out_channels=150, kernel_size=(5, 5), stride=2, padding=2),
                                  GLU(1), SwitchNorm2d(75),
                                  nn.Conv2d(in_channels=75, out_channels=200, kernel_size=(5, 5), stride=2,
                                            padding=2), GLU(1), SwitchNorm2d(100),

                                  )
        self.eacnetD = nn.Sequential(*[ECABasicBlock(100) for _ in range(1)])
        self.oooo=nn.Sequential(nn.Conv2d(in_channels=100, out_channels=2, kernel_size=(16, 16), stride=1,
                                            padding=0), GLU(1),)

    def forward(self,img):
        return  self.oooo(self.eacnetD (self.cov1(img)))


class Gx(nn.Module):
    def __init__(self, dim,eaclay):
        super().__init__()
        self.eacnet = nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=666, kernel_size=(5, 5), stride=2,
                               padding=1), GLU(1), SwitchNorm2d(333),
            nn.ConvTranspose2d(in_channels=333, out_channels=400, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(200),
            nn.ConvTranspose2d(in_channels=200, out_channels=300, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(150),
            nn.ConvTranspose2d(in_channels=150, out_channels=200, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(100),
            nn.ConvTranspose2d(in_channels=100, out_channels=6, kernel_size=(4, 4), stride=2,
                               padding=2),
            RGLU(1)

        )
    def forward(self,img_feature1,img_feature2):



        img_feature1=self.eacnet(img_feature1)
        img_feature2 = self.eacnet(img_feature2)

        img_feature1, _= img_feature1.chunk(2, dim=1)
        _, img_feature2 = img_feature2.chunk(2, dim=1)

        feature=torch.cat((img_feature1,img_feature2),dim=1)
        img=self.decode(feature)


        return img


class GAN_PFORT_DECODE(pt.LightningModule):
    def __init__(self, dim, eaclay):
        super().__init__()
        self.GG=Gx(dim=dim,eaclay=eaclay)
        self.dd=Dx(dim=dim)

        self.grad_norm = 0
        self.lrc = 0.0000


        self.automatic_optimization = False




    def forward(self,img_feature1,img_feature2):

        #
        #
        # img_feature1=self.eacnet(img_feature1)
        # img_feature2 = self.eacnet(img_feature2)
        #
        # img_feature1, _= img_feature1.chunk(2, dim=1)
        # _, img_feature2 = img_feature2.chunk(2, dim=1)
        #
        # feature=torch.cat((img_feature1,img_feature2),dim=1)
        # img=self.decode(feature)


        return self.GG(img_feature1,img_feature2)

    def configure_optimizers(self):
        optimizerG = torch.optim.AdamW(self.GG.parameters(), lr=0.0001)
        optimizerD = torch.optim.AdamW(self.dd.parameters(), lr=0.0001)
        ltG = {
            "scheduler": V3LSGDRLR(optimizerG,),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }
        ltD = {
            "scheduler": V3LSGDRLR(optimizerD, ),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }

        return [optimizerG, optimizerD], [ltG,ltD]

    def training_step(self, batch, batch_idx):
        img_f1, img_f2, t_img, img1,img2=batch
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()


##############################################
#           TD                               #
##############################################
        # loss_img = nn.L1Loss()(t_img, img)
        # img = self.forward(img_f1, img_f2)
        t_img.requires_grad_(True)
        real_output = self.dd(t_img)
        errD_real = torch.mean(torch.flatten(real_output))
        D_x = torch.flatten(real_output).mean().item()

        # Generate fake image batch with G
        fake_images = self.forward(img_f1, img_f2)

        # Train with fake
        fake_output = self.dd(fake_images)
        errD_fake = -torch.mean(torch.flatten(fake_output))
        D_G_z1 = torch.flatten(fake_output).mean().item()

        # Calculate W-div gradient penalty
        gradient_penalty = calculate_gradient_penalty(t_img, fake_images,
                                                      torch.flatten(real_output), torch.flatten(fake_output), 2, 6,
                                                      img_f1.device)

        # Add the gradients from the all-real and all-fake batches
        errDloss = errD_real + errD_fake + gradient_penalty
        # errD.backward()
        # Update D
        # self.optimizer_d.step()
###############################       ###################################
        opt_d.zero_grad()
        self.manual_backward(errDloss,retain_graph=True)
        opt_d.step()
###############################################################

        ##############################################
        #           TG                               #
        ##############################################
        # self.generator.zero_grad()
        # Generate fake image batch with G
        # fake_images = self.forward(img_f1, img_f2)
        fake_output11 = self.dd(fake_images)
        errG = torch.mean(torch.flatten(fake_output11))
        D_G_z2 = torch.flatten(fake_output11).mean().item()
        loss_img = nn.L1Loss()(t_img, fake_images)


        # if self.global_step>500:
        #     losssss = (errG + loss_img)
        # else:
        #     losssss = ( loss_img)
        losssss = ((errG + loss_img*100))/101
        opt_g.zero_grad()
        self.manual_backward(losssss)
        opt_g.step()
        sch_g.step()
        sch_d.step()


        if batch_idx%10==0 or batch_idx%11==0:
            self.wwww(batch_idx,loss_img, t_img, img1,img2,fake_images,D_G_z2,D_G_z1,D_x,gradient_penalty)


        # return loss_img

    def on_before_zero_grad(self, optimizer):
        # print(optimizer.state_dict()['param_groups'][0]['lr'],self.global_step)
        self.lrc = optimizer.state_dict()['param_groups'][0]['lr']

    def wwww(self,batch_idx,loss1, t_img, img1,img2,oimg,D_G_z2,D_G_z1,D_x,gradient_penalty):


        step=self.global_step
        # writer = tensorboard.SummaryWriter

        # writer = tensorboard
        # writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)

        writer.add_scalar('GANtrain/D_xTrue', D_x, step)
        writer.add_scalar('GANtrain/D_xfack', D_G_z1, step)
        writer.add_scalar('GANtrain/G_ganx', D_G_z2, step)
        writer.add_scalar('GANtrain/d_gradient_penalty', gradient_penalty, step)
        writer.add_scalar('train/loss1', loss1, step)
        # writer.add_scalar('train/loss2', loss2, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.add_scalar('train/lr', self.lrc, step)
        if batch_idx % 100 == 0 or batch_idx%101==0:
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

class TPFORT_DECODE(pt.LightningModule):
    def __init__(self, dim, eaclay):
        super().__init__()
        self.eacnet=nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])
        # self.eacnet2 = nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])
        self.sfa_bh = self_attention(5, 512, 8, 512, mlpdropout=0.1, attdropout=0.1 )

        # self.eacnet = Gres_modle(eaclay,dim)
        # self.eacnet = res_modle(eaclay, dim)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=666, kernel_size=(5, 5), stride=2,
                               padding=1), GLU(1), SwitchNorm2d(333),
            nn.ConvTranspose2d(in_channels=333, out_channels=400, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(200),
            nn.ConvTranspose2d(in_channels=200, out_channels=300, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(150),
            nn.ConvTranspose2d(in_channels=150, out_channels=200, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(100),
            nn.ConvTranspose2d(in_channels=100, out_channels=3, kernel_size=(6, 6), stride=2,
                               padding=3),
            nn.Softplus()
            # nn.Softmax(),
        )

        self.grad_norm = 0
        self.lrc = 0.0001
        self.emx = EMBDim(512,drop=0.1)

    def forward(self,img_feature1,img_feature2):
        # ax5 = img_feature1.detach().cpu().numpy()
        # ax6 = img_feature2.detach().cpu().numpy()
        img_feature1=self.emx(rearrange((img_feature1), 'b c h w -> b (h w) c'),0)
        img_feature2 = self.emx(rearrange((img_feature2), 'b c h w -> b (h w) c'),1)
        fuxemf=torch.cat([img_feature1,img_feature2],1)
        fff=self.sfa_bh(fuxemf)
        img_feature1, img_feature2 = fff.chunk(2, dim=1)
        img_feature1=rearrange(img_feature1, 'b (h w) c -> b c h w',h=8)
        img_feature2 = rearrange(img_feature2, 'b (h w) c -> b c h w', h=8)


        img_feature1=self.eacnet(img_feature1)
        img_feature2 = self.eacnet(img_feature2)
        # ax3 = img_feature1.detach().cpu().numpy()
        # ax4 = img_feature2.detach().cpu().numpy()

        img_feature1, _= img_feature1.chunk(2, dim=1)
        _, img_feature2 = img_feature2.chunk(2, dim=1)

        feature=torch.cat((img_feature1,img_feature2),dim=1)
        img=self.decode(feature)
        # cp1=img[1]
        # cp2 = img[0]
        # cpp=img.detach().cpu().numpy()
        # ax1=img_feature1.detach().cpu().numpy()
        # ax2=img_feature2.detach().cpu().numpy()
        pass

        return img

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        lt = {
            "scheduler": V3LSGDRLR(optimizer,),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lt
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


class LTPFORT_DECODE(pt.LightningModule):
    def __init__(self, dim, eaclay):
        super().__init__()
        self.eacnet=nn.Sequential(*[ECABasicBlock(dim*2) for _ in range(eaclay)])
        # self.eacnet2 = nn.Sequential(*[ECABasicBlock(dim) for _ in range(eaclay)])
        self.sfa_bh = self_attention(5, 512, 8, 512, mlpdropout=0.1, attdropout=0.1 )

        # self.eacnet = Gres_modle(eaclay,dim)
        # self.eacnet = res_modle(eaclay, dim)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim*2, out_channels=666, kernel_size=(5, 5), stride=2,
                               padding=1), GLU(1), SwitchNorm2d(333),
            nn.ConvTranspose2d(in_channels=333, out_channels=400, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(200),
            nn.ConvTranspose2d(in_channels=200, out_channels=300, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(150),
            nn.ConvTranspose2d(in_channels=150, out_channels=200, kernel_size=(5, 5), stride=2,
                               padding=2), GLU(1), SwitchNorm2d(100),
            nn.ConvTranspose2d(in_channels=100, out_channels=3, kernel_size=(6, 6), stride=2,
                               padding=3),
            nn.Softplus()
            # nn.Softmax(),
        )

        self.grad_norm = 0
        self.lrc = 0.0001
        self.emx = EMBDim(512,drop=0.1)

    def forward(self,img_feature1,img_feature2):
        # ax5 = img_feature1.detach().cpu().numpy()
        # ax6 = img_feature2.detach().cpu().numpy()
        img_feature1=self.emx(rearrange((img_feature1), 'b c h w -> b (h w) c'),0)
        img_feature2 = self.emx(rearrange((img_feature2), 'b c h w -> b (h w) c'),1)
        fuxemf=torch.cat([img_feature1,img_feature2],1)
        fff=self.sfa_bh(fuxemf)
        img_feature1, img_feature2 = fff.chunk(2, dim=1)
        img_feature1=rearrange(img_feature1, 'b (h w) c -> b c h w',h=8)
        img_feature2 = rearrange(img_feature2, 'b (h w) c -> b c h w', h=8)
        cimg=torch.cat([img_feature1,img_feature2],1)


        cimg=self.eacnet(cimg)
        # img_feature2 = self.eacnet(img_feature2)
        # ax3 = img_feature1.detach().cpu().numpy()
        # ax4 = img_feature2.detach().cpu().numpy()

        # img_feature1, _= img_feature1.chunk(2, dim=1)
        # _, img_feature2 = img_feature2.chunk(2, dim=1)
        #
        # feature=torch.cat((img_feature1,img_feature2),dim=1)
        img=self.decode(cimg)
        # cp1=img[1]
        # cp2 = img[0]
        # cpp=img.detach().cpu().numpy()
        # ax1=img_feature1.detach().cpu().numpy()
        # ax2=img_feature2.detach().cpu().numpy()
        pass

        return img

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        lt = {
            "scheduler": V3LSGDRLR(optimizer,),  # 调度器
            "interval": 'step',  # 调度的单位，epoch或step

            "reduce_on_plateau": False,  # ReduceLROnPlateau
            "monitor": "val_loss",  # ReduceLROnPlateau的监控指标
            "strict": False  # 如果没有monitor，是否中断训练
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lt
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
    writer = SummaryWriter("./st2_logs/", )
    modss=LTPFORT_DECODE(dim=512,eaclay=5)
    # aaaa = dastset('映射.json', 'fix1.json', './i')
    aaaa = st2_dataset('V2_dataset_stage2.hdf5','st2_rcmap','st2_map','./i/','img_maps')
    from pytorch_lightning import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=r"FORT_modle_ST2")
    checkpoint_callback = ModelCheckpoint(

        # monitor = 'val/loss',

        dirpath='./std_ckpt',

        filename='Ve2-epoch{epoch:02d}-{epoch}-{step}',

        auto_insert_metric_name=False#, every_n_epochs=20
        , save_top_k=-1,every_n_train_steps=20000

    )

    trainer = Trainer(accelerator='gpu',logger=tensorboard,max_epochs=400,callbacks=[checkpoint_callback],precision='bf16'
                      #, ckpt_path=r'C:\Users\autumn\Desktop\poject_all\Font_DL\lightning_logs\version_41\checkpoints\epoch=34-step=70000.ckpt'
                      )
    # trainer.save_checkpoint('test.pyt')
    trainer.fit(model=modss,train_dataloaders=DataLoader(dataset=aaaa,batch_size=4,shuffle=True
                                                   ,num_workers=4,prefetch_factor =16,pin_memory=True,
                                                   ))





