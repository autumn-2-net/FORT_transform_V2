import random

from io import BytesIO

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


class dastset(Dataset):
    def __init__(self,mapping,Word_path,data_path,max_tocken_len=32):
        super().__init__()
        patht=os.listdir(data_path)
        self.max_len=max_tocken_len
        pass
        with open(Word_path, 'r', encoding='utf-8') as f:
            # words = f.read()
            words=json.loads(f.read())
        with open(mapping, 'r', encoding='utf-8') as f:
            # words = f.read()
            mappi=json.loads(f.read())
        jsonnn = {}

        for i in tqdm(words):

            list1 = []
            cc = words[i]
            # list1.append(1)
            for j in cc:
                asda = mappi.get(j)
                if asda is None:
                    list1.append(47)
                    continue
                list1.append(asda)
            # list1.append(44)
            jsonnn[i] = list1
        self.wordd = jsonnn
        self.data_path=data_path
        self.cpnt={}

        for j in tqdm(self.wordd):
            tk=self.wordd[j]
            w_len = len(tk)

            maska = [1 for _ in range(w_len)]
            maskb = [0 for _ in range(self.max_len - w_len)]
            mask = maska + maskb
            tk=tk+maskb
            self.cpnt[j]=[tk,mask,w_len]
        lcslist=[]
        listss=[]
        inxlen=[]
        axc=0
        avxl=[]
        for i in tqdm(patht):
            ppp=self.get_mask_and_path(i)
            ctx=len(ppp)
            axc=axc+ctx
            lcslist=lcslist+ppp

            # listss.append(ppp)

            inxlen.append(ctx)

            avxl.append(axc)

        self.lcslist=lcslist

        self.avxl=avxl
        self.inxlen = inxlen
        self.listss=listss
        self.lenn=axc


    def get_mask_and_path(self,t_path):
        img_p=os.listdir(self.data_path+'/'+t_path)
        l=[]
        for i in img_p:
            path=self.data_path+'/'+t_path+'/'+i
            w=i.replace('.png','').strip()
            l.append([w,path])
        return l

    def get_by_idx(self,idx):

        exid=0
        il=0
        n=0
        for i in self.avxl:
            if idx <i:
                exid=idx-il
                break

            n=n+1
            il=i
        return n,exid

    def __len__(self):
        return len(self.lcslist)

    def __getitem__(self ,index):

        imgx=self.lcslist[index]
        tock,img_path=imgx[0],imgx[1]
        tocken,mask,w_len=self.cpnt[tock][0],self.cpnt[tock][1],self.cpnt[tock][2]

        maskf = w_len // 2 +1
        if random.randint(0, 20) == 1:
            maskf= w_len+10
        masktocken=tocken.copy()
        for i in range(maskf):
            if random.randint(0,500) == 1:
                break
            mxk=random.randint(0,w_len-1)
            masktocken[mxk]=43


        tocken=torch.tensor(tocken,dtype=torch.int32)
        padmask = torch.tensor(mask, dtype=torch.int32)
        masktocken = torch.tensor(masktocken, dtype=torch.int32)
        imgss = Image.open(img_path)

        transform1 = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ]
        )
        img_tensor = transform1(imgss)







        return img_tensor,tocken,padmask,masktocken









class Fdastset(Dataset):
    def __init__(self,mapping,Word_path,data_path,max_tocken_len=32):
        super().__init__()
        patht=os.listdir(data_path)
        self.max_len=max_tocken_len
        pass
        with open(Word_path, 'r', encoding='utf-8') as f:
            # words = f.read()
            words=json.loads(f.read())
        with open(mapping, 'r', encoding='utf-8') as f:
            # words = f.read()
            mappi=json.loads(f.read())
        jsonnn = {}

        for i in tqdm(words):

            list1 = []
            cc = words[i]
            # list1.append(1)
            for j in cc:
                asda = mappi.get(j)
                if asda is None:
                    list1.append(47)
                    continue
                list1.append(asda)
            # list1.append(44)
            jsonnn[i] = list1
        self.wordd = jsonnn
        self.data_path=data_path
        self.cpnt={}

        for j in tqdm(self.wordd):
            tk=self.wordd[j]
            w_len = len(tk)

            maska = [1 for _ in range(w_len)]
            maskb = [0 for _ in range(self.max_len - w_len)]
            mask = maska + maskb
            tk=tk+maskb
            self.cpnt[j]=[tk,mask,w_len]
        lcslist=[]
        listss=[]
        inxlen=[]
        axc=0
        avxl=[]
        for i in tqdm(patht):
            ppp=self.get_mask_and_path(i)
            ctx=len(ppp)
            axc=axc+ctx
            lcslist=lcslist+ppp

            # listss.append(ppp)

            inxlen.append(ctx)

            avxl.append(axc)

        self.lcslist=lcslist

        self.avxl=avxl
        self.inxlen = inxlen
        self.listss=listss
        self.lenn=axc


    def get_mask_and_path(self,t_path):
        img_p=os.listdir(self.data_path+'/'+t_path)
        l=[]

        for i in tqdm(img_p):
            path=self.data_path+'/'+t_path+'/'+i
            w=i.replace('.png','').strip()

            im=Image.open(path)
            img = np.array(im)
            l.append([w,img])
            im.close()

        return l

    def get_by_idx(self,idx):

        exid=0
        il=0
        n=0
        for i in self.avxl:
            if idx <i:
                exid=idx-il
                break

            n=n+1
            il=i
        return n,exid

    def __len__(self):
        return len(self.lcslist)

    def __getitem__(self ,index):

        imgx=self.lcslist[index]
        tock,img_path=imgx[0],imgx[1]
        tocken,mask,w_len=self.cpnt[tock][0],self.cpnt[tock][1],self.cpnt[tock][2]

        maskf = w_len // 5 + 1
        masktocken=tocken.copy()
        for i in range(maskf):
            mxk=random.randint(0,w_len-1)
            masktocken[mxk]=43


        tocken=torch.tensor(tocken,dtype=torch.int32)
        padmask = torch.tensor(mask, dtype=torch.int32)
        masktocken = torch.tensor(masktocken, dtype=torch.int32)
        imgss = img_path


        transform1 = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ]
        )
        img_tensor = transform1(imgss)







        return img_tensor,tocken,padmask,masktocken










