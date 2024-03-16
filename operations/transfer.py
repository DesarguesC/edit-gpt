import numpy as np
import torch, cv2, os, random, math, einops
from torch import nn, autocast
from paint.bgutils import target_removing
from paint.crutils import ab64
# calculate IoU between SAM & SEEM
from PIL import Image
from einops import repeat, rearrange
from paint.bgutils import refactor_mask, match_sam_box
from paint.utils import (recover_size, resize_and_pad, load_img_to_array, save_array_to_img, dilate_mask)
from operations.utils import inpaint_img_with_lama
from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything

from ldm.util import load_model_from_config, instantiate_from_config
from omegaconf import OmegaConf
import k_diffusion as K

from operations.utils import get_reshaped_img



class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)



def Transfer_Me_ip2p(
        opt, 
        input_pil: Image = None, 
        img_cfg: float = 1.5, 
        txt_cfg: float = 7.5, 
        dilate_kernel_size: int = 15,
        preloaded_model = None,
        record_history = True
    ):
    """
    -> preloded_model
        Keys in need:
            preloaded_ip2p
    """
    # 'dilate_kernel_size'  unused
    # Consider: whether mask play some roles in ip2p.
    if preloaded_model is None:
        config = OmegaConf.load(opt.ip2p_config)
        ip2p_model = load_model_from_config(config, opt.ip2p_ckpt, opt.vae_ckpt).eval()
        if torch.cuda.is_available(): ip2p_model = ip2p_model.cuda()
        ip2p_wrap = K.external.CompVisDenoiser(ip2p_model)
        model_wrap_cfg = CFGDenoiser(ip2p_wrap)
        null_token = ip2p_model.get_learned_conditioning([""])
    else:
        preloaded_ip2p_dict = preloaded_model['preloaded_ip2p']
        ip2p_model = preloaded_ip2p_dict['model'] # cuda convertion has done
        if torch.cuda.is_available(): ip2p_model = ip2p_model.cuda()
        ip2p_wrap = preloaded_ip2p_dict['wrap']
        model_wrap_cfg = CFGDenoiser(ip2p_wrap)
        null_token = preloaded_ip2p_dict['null_token']

    seed = random.randint(0, 1e5) if opt.seed is None else opt.seed
    opt, img_pil = get_reshaped_img(opt, input_pil)
    # note: difference between ImageOps.fit and .resize ?
    t = 0
    name_ = f'./{opt.base_folder}/{opt.out_name}'
    if opt.edit_txt == "":
        while os.path.isfile(f'{name_}-{t}.jpg'): t += 1
        img_pil.save(f'{name_}-{t}.jpg')
        return
    
    with torch.no_grad(), autocast("cuda"), ip2p_model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [ip2p_model.get_learned_conditioning([opt.edit_txt])]
        img = 2 * torch.tensor(np.array(img_pil)).float() / 255 - 1
        img = rearrange(img, "h w c -> 1 c h w").to(ip2p_model.device)
        cond["c_concat"] = [ip2p_model.encode_first_stage(img).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = ip2p_wrap.get_sigmas(opt.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": txt_cfg,
            "image_cfg_scale": img_cfg,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = ip2p_model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    
    name_ = f'./{opt.base_folder}/{opt.out_name}' if record_history else f'./{opt.out_dir}/Transfer/trans'
    while os.path.isfile(f'{name_}-{t}.jpg'): t += 1
    edited_image.save(f'{name_}-{t}.jpg')
    return edited_image # pil
    

