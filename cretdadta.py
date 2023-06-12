import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

from fontTools.ttLib import TTFont
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm


class data_create():
    def __init__(self,map_path,w_path,ttf_path):
        self.map_path = map_path
        self.w_path = w_path
        self.ttf_path = ttf_path

        with open(w_path, 'r', encoding='utf-8') as f:
            # words = f.read()
            words=json.loads(f.read())
        with open(map_path, 'r', encoding='utf-8') as f:
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
            list1.append(44)
            jsonnn[i] = list1
        self.wordd = jsonnn
        self.tttfp=glob.glob(ttf_path+'/**.ttf')
        # self.tttfp=os.listdir(ttf_path)
        pass

    def get_anc_c(self,):
        # font = TTFont(path_ttfs)
        # # 输出的uniMap是一个字典，key代表的unicode的int值，value代表unicode的名字
        # uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
        # # print(ord(tocken) in uniMap.keys())
        # tfss = ord(tocken) in uniMap.keys()
        # text_size = 255  # text_size 是字号
        # font = ImageFont.truetype(self.ttf_path + i, text_size)
        # if tfss:
        #     imggg = self.get_img(ttstk, tocken)
        #     return tocken, imggg
        rree=[]
        for i in tqdm(self.tttfp):
            rree.append(( i,'./i',i.split('\\')[-1],self.wordd))
        # for i in self.tttfp:
        with ProcessPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(make_and_w,rree), desc='Preprocessing', total=len(self.tttfp)))

            # make_and_w(i,'./i',i.split('\\')[-1],self.wordd)


            # for j in self.wordd:
            #     print(j)
            # pass
def make_and_w(cx):
    f_path, out_path, ttfname, wm=cx
    # print(f_path)
    font = TTFont(f_path)
    uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
    text_size = 255  # text_size 是字号
    font = ImageFont.truetype(f_path, text_size)

    if not os.path.exists(out_path+'/'+ttfname+'/'):
        os.makedirs(out_path+'/'+ttfname+'/')

    for i in wm:
        tfss = ord(i) in uniMap.keys()
        if not tfss:
            continue

        x, y = font.getsize(i)

        y = max(y, 256)
        x = max(x, 256)
        cavv = Image.new('RGB', (x, y), (255, 255, 255))
        cavvii = Image.new('RGB', (x, y), (255, 255, 255))
        ddd = ImageDraw.Draw(cavv)
        ddd.text((0, 0), i, font=font, fill='#000000')

        img=cavv.resize((256, 256), Image.Resampling.LANCZOS)

        if img ==cavvii:  #判断白字
            # print(i)
            continue

            # raise Exception('eeeeeeeeeeeeeeeeeee')

        img.save(out_path+'/'+ttfname+'/'+i+'.png',)
        pass



class data_test():
    def __init__(self,map_path,w_path,ttf_path):
        self.map_path = map_path
        self.w_path = w_path
        self.ttf_path = ttf_path

        with open(w_path, 'r', encoding='utf-8') as f:
            # words = f.read()
            words=json.loads(f.read())
        with open(map_path, 'r', encoding='utf-8') as f:
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
            list1.append(44)
            jsonnn[i] = list1
        self.wordd = jsonnn
        self.tttfp=glob.glob(ttf_path+'/**.ttf')
        # self.tttfp=os.listdir(ttf_path)
        pass

    def get_anc_c(self,):
        # font = TTFont(path_ttfs)
        # # 输出的uniMap是一个字典，key代表的unicode的int值，value代表unicode的名字
        # uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
        # # print(ord(tocken) in uniMap.keys())
        # tfss = ord(tocken) in uniMap.keys()
        # text_size = 255  # text_size 是字号
        # font = ImageFont.truetype(self.ttf_path + i, text_size)
        # if tfss:
        #     imggg = self.get_img(ttstk, tocken)
        #     return tocken, imggg
        rree=[]
        for i in tqdm(self.tttfp):
            rree.append(( i,'./i',i.split('\\')[-1],self.wordd))
        # for i in self.tttfp:
        rree.reverse()
        for i in rree:
            print(i[2])
            make_and_wT(i)
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     list(tqdm(executor.map(make_and_w,rree), desc='Preprocessing', total=len(self.tttfp)))

def make_and_wT(cx):
    f_path, out_path, ttfname, wm=cx
    # print(f_path)
    font = TTFont(f_path)
    uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
    text_size = 255  # text_size 是字号
    font = ImageFont.truetype(f_path, text_size)

    if not os.path.exists(out_path+'/'+ttfname+'/'):
        os.makedirs(out_path+'/'+ttfname+'/')


    for i in tqdm(wm):
        tfss = ord(i) in uniMap.keys()
        if not tfss:
            continue

        x, y = font.getsize(i)

        y = max(y, 256)
        x = max(x, 256)
        cavv = Image.new('RGB', (x, y), (255, 255, 255))
        cavvii = Image.new('RGB', (x, y), (255, 255, 255))
        ddd = ImageDraw.Draw(cavv)
        ddd.text((0, 0), i, font=font, fill='#000000')

        img=cavv.resize((256, 256), Image.Resampling.LANCZOS)

        if img ==cavvii:  #判断白字
            # print(i)
            continue

            # raise Exception('eeeeeeeeeeeeeeeeeee')

        # img.save(out_path+'/'+ttfname+'/'+i+'.png',)
        # pass
        #

if __name__=='__main__':
    # aas=data_test(w_path='fix1.json', map_path='映射.json', ttf_path='./tds/')
    aas = data_create(w_path='fix1.json', map_path='映射.json', ttf_path='./tds/')
    aas.get_anc_c()