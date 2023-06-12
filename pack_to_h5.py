import json
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
#
# im = Image.open('./i/Aa剑豪体.ttf/丢.png')
# img = np.array(im)
# pass
#
#
#
#
#
#
#
# with h5py.File('myfile.hdf5','r') as f:
#     ppp=f["ints"][:]
#     pass

class dastset():
    def __init__(self,mapping,Word_path,data_path,max_tocken_len=32):

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
        n=0
        cpx={}
        with h5py.File('myfile.hdf5', 'w') as f:

            for i in tqdm(patht):
                ppp=self.get_mask_and_path(i)
                for ii in tqdm(ppp):
                    cpx[str(n)]=ii[0]
                    im = Image.open(ii[1])
                    img = np.array(im)
                    f.create_dataset(str(n), data=img)



                    n=+n+1

        with open('cpx.json','w',encoding='utf8') as x:
            x.write(json.dumps(cpx,ensure_ascii=False))


    def get_mask_and_path(self,t_path):
        img_p=os.listdir(self.data_path+'/'+t_path)
        l=[]
        for i in img_p:
            path=self.data_path+'/'+t_path+'/'+i
            w=i.replace('.png','').strip()
            l.append([w,path])
        return l

if __name__=='__main__':
    dastset('映射.json', 'fix1.json', './i')



















