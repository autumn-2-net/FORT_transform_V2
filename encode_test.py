import json
import os

import h5py

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from Tencode import PFORT_encode

class dastsetss(Dataset):
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
        self.cnnax=[]
        for i in tqdm(patht):
            # print(i)
            self.cnnax.append(i)
            ppp=self.get_mask_and_path(i)
            ctx=len(ppp)
            axc=axc+ctx
            # lcslist=lcslist+ppp

            listss.append(ppp)

            inxlen.append(ctx)

            avxl.append(axc)

        self.lcslist=lcslist

        self.avxl=avxl
        self.inxlen = inxlen
        self.listss=listss
        self.lenn=axc


    def get_nme(self):
        n=0
        dd={}
        for i in self.cnnax:
            dd[str(n)]=i
            n=n+1
        return dd


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
        return self.lenn

    def __getitem__(self ,index):


        xxxxx=self.get_by_idx(index)
        imgx=self.listss[xxxxx[0]][xxxxx[1]]
        tock,img_path=imgx[0],imgx[1]
        tocken,mask,w_len=self.cpnt[tock][0],self.cpnt[tock][1],self.cpnt[tock][2]





        tocken=torch.tensor(tocken,dtype=torch.int32)
        padmask = torch.tensor(mask, dtype=torch.int32)

        imgss = Image.open(img_path)

        transform1 = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ]
        )
        img_tensor = transform1(imgss)







        return img_tensor,tocken,padmask,tock,xxxxx[0],xxxxx[1]








if __name__=='__main__':

    modss=PFORT_encode(ATTlays=6, bhlay=9, imglay=5, dim=512, heads=8, inner_dim=512, out_dim=48, pos_emb_drop=0.1,
                         mlpdropout=0.05, attdropout=0.05)
    # aaaa = dastset('映射.json', 'fix1.json', './i')
    modss=modss.load_from_checkpoint('./post_LN/V6-epoch06-6-240000.ckpt',ATTlays=6, bhlay=9, imglay=5, dim=512, heads=8, inner_dim=512, out_dim=48, pos_emb_drop=0.1,
                         mlpdropout=0.05, attdropout=0.05)
    modss.eval().cuda()
    aaaa=dastsetss('映射.json', 'fix1.json', './i')

    pppoio={}
    pppoioF = {}
    with open("img_maps",'w',encoding="utf-8" ) as f:
        f.write(json.dumps(aaaa.get_nme(),ensure_ascii=False))
    with h5py.File('V2_dataset_stage2.hdf5', 'w') as f:


        for i in tqdm(DataLoader(dataset=aaaa,batch_size=32,shuffle=False
                                                       ,#num_workers=4,prefetch_factor =16,pin_memory=True,
                                                       )):


            featrue_img=modss.encode(i[0].cuda(),i[1].cuda(),bhmask=i[2].cuda())

            for cpx in zip(featrue_img,i[3],i[4].numpy(),i[5].numpy()):
                featrue_img,ci,ftt,idx=cpx
                if pppoio.get(str(ftt)) is None:
                    pppoio[str(ftt)]={}
                if pppoioF.get(str(ftt)) is None:
                    pppoioF[str(ftt)]={}

                pppoio[str(ftt)][str(idx)]=str(ci)
                pppoioF[str(ftt)][str(ci)] = str(idx)
                f.create_dataset(f'{str(ftt)}_{str(idx)}', data=featrue_img.detach().cpu().numpy(), compression="gzip",compression_opts=9#('ec'|'nn', even integer 0-32)
                                 )

                pass

    with open('st2_rcmap','w',encoding='utf8') as f:
        f.write(json.dumps(pppoio,ensure_ascii=False))
    with open('st2_map','w',encoding='utf8') as f:
        f.write(json.dumps(pppoioF,ensure_ascii=False))






