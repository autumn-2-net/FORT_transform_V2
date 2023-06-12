import random



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
        for i in words:

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

        for j in self.wordd:
            tk=self.wordd[j]
            w_len = len(tk)

            maska = [1 for _ in range(w_len)]
            maskb = [0 for _ in range(self.max_len - w_len)]
            mask = maska + maskb
            tk=tk+maskb
            self.cpnt[j]=[tk,mask]

        listss=[]
        inxlen=[]
        axc=0
        avxl=[]
        for i in patht:
            ppp=self.get_mask_and_path(i)
            ctx=len(ppp)
            axc=axc+ctx
            listss.append([ppp])
            inxlen.append(ctx)
            avxl.append(axc)
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

        il=0
        n=0
        for i in self.avxl:
            if idx <i:
                exid=idx-il
                break

            n=n+1
            il=i
        return n,exid














