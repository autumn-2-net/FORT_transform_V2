
from base_modle.positemb import RelPositionalEncoding
import torch

from torch import nn

























class EMBDim(nn.Module):
    def __init__(self,dim,max_tochen_len=128,postlen=65,embt='REL',drop=0.0):
        super().__init__()
        self.type_emb=nn.Embedding(2,dim) #0 img 1bh
        self.posTYPE=embt
        self.position = nn.Embedding(num_embeddings=postlen, embedding_dim=dim)


        self.RELposition=RelPositionalEncoding(dim,drop)
    def get_post(self,x,mask=None):
        if self.posTYPE == 'REL':
            x=self.RELposition(x)
            if mask is not None:
                x=x.masked_fill(mask.unsqueeze(2)==0,0)
            return x
        elif self.posTYPE == 'Learn':

            L=x.size()[2]
            p=torch.tensor([i for i in range(L)]).unsqueeze(0).unsqueeze(0).to(x)
            pos=self.position(p)
            if mask is not None:
                out=(x+pos).masked_fill(mask==0,0)
            return out
        else:
            raise Exception('eeeeee')

    def forward(self,x,t,mask=None):
        if t==1:

            img=x+self.type_emb(torch.tensor([1], dtype=torch.int32,device=x.device)).unsqueeze(0).to(x)
            return self.get_post(img,mask=mask)
        if t == 0:
            img=x+self.type_emb(torch.tensor([0], dtype=torch.int32,device=x.device)).unsqueeze(0).to(x)
            return self.get_post(img, mask=mask)
