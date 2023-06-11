# Optionally use the context manager to ensure one of the fused kerenels is run
import torch
import torch.nn.functional as F
from torch import nn

query = torch.rand(4, 8, 8, 64, dtype=torch.float16, device="cuda")
key = torch.ones(4, 8, 4, 64, dtype=torch.float16, device="cuda")
value = torch.rand(4, 8, 4, 64, dtype=torch.float16, device="cuda")
# with torch.backends.cuda.sdp_kernel(enable_math=False):
attn=torch.tensor([[True, True, True,False],[True, True, True,False],[True, True, True,False],[False, False, False,False]]).unsqueeze(1).unsqueeze(1)
atn =~ attn
d =(query@key.transpose(-2, -1)).cpu().masked_fill(~ attn, -float('inf')).numpy()
a =F.scaled_dot_product_attention(query,key,value,attn_mask= attn.cuda(), )
a
c=a.cpu().numpy()
c


ascs=nn.Embedding(3,512,padding_idx=0)
e=ascs(torch.tensor([1]))
es=ascs(torch.tensor([0]))
e