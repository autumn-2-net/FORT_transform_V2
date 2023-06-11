import math

import torch
import torch.nn as nn

from torch.nn.functional import cross_entropy

# from memonger import SublinearSequential #内存压缩


class main_config():
    n_img_embd =512 #图片特征维度
    n_bh_embd = 512  # 笔画特征维度
    n_head = 16  #8头吧？
    n_embd =n_img_embd
    n_vebs =32 #笔画长度
    n_vebs_long =48
    n_img_s =64 #图像长度
    img_bert_lay =6
    bh_bert_lay =4
    co_att_lay =5
    embd_pdrop =0.1
    LR=0.0003



class MultiheadSelfAttention(nn.Module):

    # nn.MultiheadAttention()

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        # self.attn_drop = nn.Dropout(config.attn_pdrop)
        # self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)) #掩码 没必要bert
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None,att_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print('a')
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # if att_mask is not None: #根据原文mask的只有 图像部分 当输入补pad的时候 因为图像不会被mask 使用不需要？
        #     maskt =att_mask.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)@ k.transpose(-2, -1)
        #     one = torch.ones_like(maskt)
        #     maskt =torch.where(maskt > 0.0, one, maskt)
        #     maskt = torch.where(maskt < 0.0, one, maskt)
        #     maskt = maskt*-10000
        #     att =att+maskt

        att = torch.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_drop(self.proj(y))
        y=self.proj(y)
        return y


class BERT_Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadSelfAttention(config)
        # self.attn = SublinearSequential(self.attn)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            # nn.Dropout(config.embd_pdrop),
        )


        # self.mlp = SublinearSequential(*list(self.mlp.children()))  #减小400m

    def forward(self, x,att_mask):

        x = x + self.attn(self.ln1(x),att_mask=att_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class bert_modle(nn.Module):
    def __init__(self,config,layers=4):
        super().__init__()
        # self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([BERT_Block(config=config) for _ in range(layers)])
        # self.blocks = nn.Sequential(*[BERT_Block(config=config) for _ in range(layers)])
        # self.blocks =SublinearSequential(*list(self.blocks.children()))

    def forward(self, x,att_mask=None):
        for i in self.blocks:
            x =i(x,att_mask)
        return x
















class co_attentional(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_bh_embd % config.n_head == 0
        assert config.n_img_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.value_img_to_bh = nn.Linear(config.n_img_embd, config.n_img_embd)
        self.key_img_to_bh = nn.Linear(config.n_img_embd, config.n_img_embd)
        self.query_img = nn.Linear(config.n_img_embd, config.n_img_embd)


        self.value_bh_to_img  = nn.Linear(config.n_bh_embd, config.n_bh_embd)
        self.key_bh_to_img = nn.Linear(config.n_bh_embd, config.n_bh_embd)
        self.query_bh = nn.Linear(config.n_bh_embd, config.n_bh_embd)

        # regularization
        # self.attn_drop = nn.Dropout(config.attn_pdrop)
        # self.resid_drop = nn.Dropout(config.embd_pdrop)
        # output projection
        self.proj_img = nn.Linear(config.n_img_embd, config.n_img_embd) #这就一个映射层
        self.proj_bh = nn.Linear(config.n_bh_embd, config.n_bh_embd)
        # nn.LayerNorm()
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)) #掩码 没必要bert
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_img, y_bh,attention_mask=None,img_attention_mask=None):
        B_img, T_img, C_img = x_img.size()
        B_bh, T_bh, C_bh = y_bh.size()
        #
        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k = self.key(x_img).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # q = self.query(x_img).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # v = self.value(x_img).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # print('a')
        q_img =self.query_img(x_img).view(B_img, T_img, self.n_head, C_img // self.n_head).transpose(1, 2)
        k_f_img=self.key_img_to_bh(x_img).view(B_img, T_img, self.n_head, C_img // self.n_head).transpose(1, 2)
        v_f_img = self.value_img_to_bh(x_img).view(B_img, T_img, self.n_head, C_img // self.n_head).transpose(1, 2)

        q_bh = self.query_bh(y_bh).view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)
        k_f_bh = self.key_bh_to_img(y_bh).view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)
        v_f_bh = self.value_bh_to_img(y_bh).view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)

        att_img=(q_img @ k_f_bh.transpose(-2, -1))* (1.0 / math.sqrt(k_f_bh.size(-1)))
        # if attention_mask is not None: #根据原文mask的只有 图像部分 当输入补pad的时候 因为图像不会被mask 使用不需要？，我超 znm根本就mask不上去
        #     att_img =att_img+attention_mask.view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)
        if attention_mask is not None: #根据原文mask的只有 图像部分 当输入补pad的时候 因为图像不会被mask 使用不需要？
            maskt =attention_mask.view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)
            maskt =q_img @ maskt.transpose(-2, -1)
            one = torch.ones_like(maskt)
            maskt =torch.where(maskt > 0.0, one, maskt)
            maskt = torch.where(maskt < 0.0, one, maskt)
            maskt = maskt*-10000
            att_img =att_img+maskt
        att_img = torch.softmax(att_img, dim=-1)
        out_img =att_img @ v_f_bh
        out_img = out_img.transpose(1, 2).contiguous().view(B_img, T_img, C_img)
        out_img =self.proj_img(out_img)

        att_bh =(q_bh @ k_f_img.transpose(-2, -1))* (1.0 / math.sqrt(k_f_img.size(-1)))
        # if attention_mask is not None: #根据原文mask的只有 图像部分 当输入补pad的时候 因为图像不会被mask 使用不需要？，我超 znm根本就mask不上去
        #     att_bh =att_bh+attention_mask.view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)

        # if attention_mask is not None: #根据原文mask的只有 图像部分 当输入补pad的时候 因为图像不会被mask 使用不需要？
        #     maskt =attention_mask.view(B_bh, T_bh, self.n_head, C_bh // self.n_head).transpose(1, 2)
        #     maskt =maskt @k_f_img.transpose(-2, -1)
        #     one = torch.ones_like(maskt)
        #
        #     maskt =torch.where(maskt > 0.0, one, maskt)
        #     maskt = torch.where(maskt < 0.0, one, maskt)
        #     maskt = maskt*-10000
        #     sssss =maskt.detach().cpu().clone().numpy()
        #     att_bh =att_bh+maskt
        att_bh = torch.softmax(att_bh, dim=-1)
        out_bh = att_bh @ v_f_img
        out_bh = out_bh.transpose(1, 2).contiguous().view(B_bh, T_bh, C_bh)
        out_bh = self.proj_bh(out_bh)


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # att = torch.softmax(att, dim=-1)
        # # att = self.attn_drop(att)
        # y_bh = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y_bh = y_bh.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_drop(self.proj(y))
        # y_bh = self.proj(y_bh)


        return out_img,out_bh


class co_encode_lay(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.co_att =co_attentional(config)
        self.img_att = MultiheadSelfAttention(config)
        self.bh_att = MultiheadSelfAttention(config)
        self.LN_img=nn.LayerNorm(config.n_img_embd, eps=1e-12)
        self.LN_bh = nn.LayerNorm(config.n_bh_embd, eps=1e-12)

        self.LN_img2 = nn.LayerNorm(config.n_img_embd, eps=1e-12)
        self.LN_bh2 = nn.LayerNorm(config.n_bh_embd, eps=1e-12)

        self.LN_img3 = nn.LayerNorm(config.n_img_embd, eps=1e-12)
        self.LN_bh3 = nn.LayerNorm(config.n_bh_embd, eps=1e-12)

        self.mlp_bh = nn.Sequential(
            nn.Linear(config.n_bh_embd, 4 * config.n_bh_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_bh_embd, config.n_bh_embd),
            # nn.Dropout(config.embd_pdrop),
        )
        self.mlp_img = nn.Sequential(
            nn.Linear(config.n_img_embd, 4 * config.n_img_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_img_embd, config.n_img_embd),
            # nn.Dropout(config.embd_pdrop),
        )



    def forward(self,input1,att_mask_bh=None,att_mask_img=None):
        img, bh =input1
        x_img,y_bh =self.co_att(img,bh,attention_mask=att_mask_bh)
        x_img=x_img+img
        x_img = self.LN_img(x_img) #归一
        x1 =x_img
        x_img =self.img_att(x_img)
        x_img = x_img + x1
        x_img = self.LN_img2(x_img)  # 归一
        x2 =x_img
        x_img =self.mlp_img(x_img)
        x_img = x_img + x2
        x_img = self.LN_img3(x_img)


        y_bh=y_bh + bh
        y_bh=self.LN_bh(y_bh)
        y1 =y_bh
        y_bh = self.bh_att(y_bh,att_mask=att_mask_bh)
        y_bh = y_bh + y1
        y_bh = self.LN_bh2(y_bh)  # 归一
        y2 =y_bh
        y_bh =self.mlp_bh(y_bh)
        y_bh =y_bh + y2
        y_bh = self.LN_bh3(y_bh)

        return x_img,y_bh

class co_encode_modle(nn.Module):
    def __init__(self,config,layers=5):
        super().__init__()
        self.blocks = nn.ModuleList([co_encode_lay(config=config) for _ in range(layers)])
        # self.blocks = SublinearSequential(*list(self.blocks.children()))

    def forward(self, x,y,att_mask=None,att_mask_img=None):
        for i in self.blocks:
            x,y =i((x,y), att_mask,att_mask_img)
        return x,y
























