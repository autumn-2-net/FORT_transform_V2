import random

from io import BytesIO

import h5py
import numpy as np
import pytorch_lightning as pyl
import torch
import torch.nn as nn
import  json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from fontTools.ttLib import TTFont
from PIL import ImageFont, Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm


class st2_dataset(Dataset):
    def __init__(self,h5p,remapp,mapp,st1p,st1map):
        super().__init__()
        self.st1p=st1p
        with open(st1map,'r',encoding='utf8') as f:
            self.st1map = json.loads(f.read())
        self.lans=h5p



        self.h5p = None
        with open(remapp,'r',encoding='utf8' ) as f:
            rmap=json.loads(f.read())

        with open(mapp,'r',encoding='utf8' ) as f:
            map=json.loads(f.read())
        self.rmap=rmap
        self.map=map

        lxt=len(rmap)
        self.lxt=lxt
        pass
        cpnt = []
        lenxc=0
        self.avxl1=[]
        for i in range(lxt):
            cpxta=len(rmap[str(i)])

            cpnt.append(cpxta)
            lenxc=lenxc+cpxta
            self.avxl1.append(lenxc)

        self.avxl = cpnt
        self.lcslist=lenxc

        # for i in rmap:
        #     print(i)
    def __len__(self):
        return self.lcslist
    def get_by_idx(self,idx):

        exid=0
        il=0
        n=0
        for i in self.avxl1:
            if idx <i:
                exid=idx-il
                break

            n=n+1
            il=i
        return n,exid

    def get_enten_img(self,b_id):
        transform1 = transforms.Compose([
            transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
        ]
        )

        img_ld1=self.h5p[f'{str(b_id[0])}_{str(b_id[1])}'][()]
        imwx = self.rmap[str(b_id[0])][str(b_id[1])]

        for i in range(6):
            st2_d = random.randint(0, self.lxt - 1)
            if st2_d!=b_id[0]:
                fullL=self.avxl[st2_d]
                ctap = random.sample(range(fullL), 2)
                img_ld2 = self.h5p[f'{str(st2_d)}_{str(ctap[0])}'][()]
                if self.map[str(st2_d)].get(imwx) is None:
                    continue

                imgs_tg=transform1(self.img_get(str(st2_d),str(self.map[str(st2_d)].get(imwx))))

                imgs_1 = transform1(self.img_get(str(b_id[0]), str(b_id[1])))
                imgs_2= transform1(self.img_get(str(st2_d), str(ctap[0])))

                return torch.tensor(img_ld1),torch.tensor(img_ld2),imgs_tg,imgs_1,imgs_2
        # print(imwx)
        imgs_1 = transform1(self.img_get(str(b_id[0]), str(b_id[1])))
        return torch.tensor(img_ld1),torch.tensor(img_ld1),imgs_1,imgs_1,imgs_1



    def img_get(self,G,idx):

        imw = self.rmap[str(G)][str(idx)]
        namG=self.st1map[str(G)]
        ctxp=f'{self.st1p}/{namG}/{imw}.png'
        imgss = Image.open(ctxp)
        return imgss




    def __getitem__(self ,index):
        if self.h5p is None:
            self.h5p=h5py.File(self.lans, 'r')
        base_id=self.get_by_idx(index)
        il1,il2,it,im1,im2=self.get_enten_img(base_id)
        return il1,il2,it,im1,im2









if __name__=='__main__':
    ctx=st2_dataset()
