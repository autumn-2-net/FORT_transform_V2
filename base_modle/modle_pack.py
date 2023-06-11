import math

import torch
import torch.nn as nn
from base_modle.bert_modle import co_encode_modle,bert_modle
from  base_modle.cov_encode import cov_encode


class main_encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.img_bert =bert_modle(config=config,layers=config.img_bert_lay)
        self.bh_bert = bert_modle(config=config, layers=config.bh_bert_lay)
        self.co_attentional =co_encode_modle(config=config,layers=config.co_att_lay)

    def forward(self,img,bh,att_mask=None,img_attention_mask=None):
        imgg =self.img_bert(img)
        if att_mask is not None:
            att_mask =att_mask.to(bh.device)
        bhh =self.bh_bert(bh,att_mask)
        out_img,out_bh = self.co_attentional(imgg,bhh,att_mask)
        return out_img, out_bh

class world_img_embedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=48,embedding_dim=config.n_img_embd,padding_idx=0) #词嵌入
        self.position = nn.Embedding(num_embeddings=65,embedding_dim=config.n_img_embd,padding_idx=64)  # 1 -64
        self.type1 = nn.Embedding(num_embeddings=3,embedding_dim=config.n_img_embd,padding_idx=2)  # 0-img,1-bh两类
        self.LayerNorm_bh = nn.LayerNorm(config.n_embd, eps=1e-12)
        self.LayerNorm_img = nn.LayerNorm(config.n_embd, eps=1e-12)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x,type):
        seq_length = x.size(1)
        bach_size = x.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(torch.randn((bach_size,seq_length), device=x.device))
        if type == 0: #img
            token_type_ids = torch.zeros_like(torch.arange(seq_length, dtype=torch.long, device=x.device), dtype=torch.long, device=x.device)
            token_type_ids = token_type_ids.unsqueeze(0).expand_as(torch.randn((bach_size,seq_length), device=x.device))
            position =self.position(position_ids)
            typee =self.type1(token_type_ids)
            return self.LayerNorm_img(x+position+typee)



        if type == 1:
            token_type_ids = torch.ones_like(torch.randn((bach_size,seq_length), device=x.device)).long()
            position = self.position(position_ids)
            typee = self.type1(token_type_ids)
            v_embedding = self.embedding(x)
            return self.LayerNorm_bh(v_embedding + position + typee)

def make_mask(mask,bh,config):
    seq_length = bh.size(1)
    bach_size = bh.size(0)

    mask1 =mask.unsqueeze(2).expand((bach_size,seq_length,config.n_embd),).to(bh.device)
    mask1 =(1 -mask1) *-10000
    return mask1.type(torch.FloatTensor)


class part_of_main_modle(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.cov_encode=cov_encode()
        self.embedding =world_img_embedding(config=config)
        self.main_encoder=main_encoder(config=config)
        self.config=config

    def forward(self, img,bh,att_mask=None,att_mask_img=None):

        x=self.cov_encode(img)
        img_cov =x.view(x.size(0),self.config.n_img_embd,64).transpose(1, 2)
        # img_cov = img_cov.view((512,64)).transpose(0, 1)
        img_emb =self.embedding(img_cov,0)
        bh_embedding =self.embedding(bh,1)
        if att_mask is not None:
            # print('a')
            mask=make_mask(bh=bh,mask=att_mask,config=self.config)
            mask =mask.to(bh.device)
        else:
            mask =None
        return self.main_encoder(img_emb,bh_embedding,mask)

