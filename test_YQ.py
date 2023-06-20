import numpy as np
import torch
from PIL import Image
from einops import rearrange

from YQgan.VQ_gan import VQModel





if __name__=='__main__':
    vvvv =VQModel(  ddconfig={'double_z': False ,'z_channels': 4 ,'resolution': 256 ,'in_channels': 3 ,'out_ch': 3 ,'ch': 128
              ,'ch_mult' :[1 ,2 ,2 ,4] ,'num_res_blocks': 2 ,'attn_resolutions' :[32] ,'dropout': 0.0},
               lossconfig=0,
               n_embed=16384,
               embed_dim=4,
               ckpt_path=None,
               ignore_keys=[],
               image_key="image",
               colorize_nlabels=None,
               monitor=None,
               remap=None,
               sane_index_shape=False, )

    vvvv = vvvv.load_from_checkpoint(r'C:\Users\autumn\Downloads\新建文件夹 (19)/model.ckpt',
                                     ddconfig={'double_z': False, 'z_channels': 4, 'resolution': 256, 'in_channels': 3,
                                               'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 2, 4], 'num_res_blocks': 2,
                                               'attn_resolutions': [32], 'dropout': 0.0},
                                     lossconfig=0,
                                     n_embed=16384,
                                     embed_dim=4,
                                     ckpt_path=None,
                                     ignore_keys=[],
                                     image_key="image",
                                     colorize_nlabels=None,
                                     monitor=None,
                                     remap=None,
                                     sane_index_shape=False, ).eval()

    # image = Image.open('ssss.png')
    image = Image.open('i/Aa剑豪体.ttf/丢.png')
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    # image = self.preprocessor(image=image)["image"]
    image = (image / 127.5 - 1.0).astype(np.float32)
    # image = (image / 255).astype(np.float32)

    # aaa=rearrange(torch.tensor(image), 'w h x -> x w h')
    aaa = rearrange(torch.tensor(image), 'w h x -> x h w')
    r=vvvv(aaa.unsqueeze(0).cuda())#.transpose(1,2,0)
    # rrr=((rearrange(r[0].detach().cpu().squeeze(0), 'x w h -> w h x').numpy()+1)*127.5).astype(np.uint8)
    rrr = ((rearrange(r[0].detach().cpu().squeeze(0), 'x h w -> w h x').numpy() + 1) * 127.5).astype(np.uint8)

    # rrr = ((rearrange(r[0].detach().cpu().squeeze(0), 'x w h -> w h x').numpy() ) *255).astype(np.uint8)
    Image.fromarray(rrr).save('2561ss22.png')
    pass

