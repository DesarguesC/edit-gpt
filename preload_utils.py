from omegaconf import OmegaConf
import k_diffusion as K
from PIL import Image
from einops import repeat, rearrange
import numpy as np
import torch, cv2, os, random, math
from torch import nn, autocast

from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything
from ldm.util import load_model_from_config, instantiate_from_config
from ldm.inference_base import *
from paint.control import get_adapter, get_adapter_feature, get_style_model, process_style_cond

from pldm.models.diffusion.ddim import DDIMSampler
from pldm.models.diffusion.plms import PLMSSampler

from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux.lineart import LineartDetector
from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize


def preload_ip2p(opt):
    config = OmegaConf.load(opt.ip2p_config)
    ip2p_model = load_model_from_config(config, opt.ip2p_ckpt, opt.vae_ckpt).eval()
    if torch.cuda.is_available(): ip2p_model = ip2p_model.cuda()
    ip2p_wrap = K.external.CompVisDenoiser(ip2p_model)
    null_token = ip2p_model.get_learned_conditioning([""])

    return {
        'model': ip2p_model,
        'wrap': ip2p_wrap,
        'null_token': null_token,
    }

def preload_XL_generator(opt):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(f"{opt.XL_base_path}/stabilityai/stable-diffusion-xl-base-1.0", \
                                                torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
    pipe.to("cuda")

    return {
        'pipe': pipe
    }

def preload_XL_adapter_generator(opt):
    
    opt.XL_base_path = opt.XL_base_path.strip('/')
    # load adapter
    adapter = T2IAdapter.from_pretrained(
        f"{opt.XL_base_path}/TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16, varient="fp16", local_files_only=True
    ).to("cuda") if opt.linear else T2IAdapter.from_pretrained(
        f"{opt.XL_base_path}/TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16", local_files_only=True
    ).to("cuda")
    # load euler_a scheduler
    model_id = f'{opt.XL_base_path}/stabilityai/stable-diffusion-xl-base-1.0'
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(f"{opt.XL_base_path}/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, local_files_only=True)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", local_files_only=True
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    detector = LineartDetector.from_pretrained(f"{opt.XL_base_path}/lllyasviel/Annotators").to("cuda") if opt.linear else None
    # TODO: depth detector needs to be added

    return {
        'pipe': pipe, 
        'detector': detector
    }

def preload_v1.5_generator(opt):
    # for both v1.5 and v1.5_adapter
    sd_model, sd_sampler = get_sd_models(opt)
    if opt.example_type == 'v1.5_adapter':
        print('-'*9 + 'Generating via Style Adapter (depth)' + '-'*9)
        adapter, cond_model = get_adapter(opt, cond_type='depth'), get_depth_model(opt)
        print(f'BEFORE: cond_img.size = {ori_img.size}')
        cond = process_depth_cond(opt, ori_img, cond_model) # not a image
        print(f'cond.shape = {cond.shape}, cond_mask.shape = {cond_mask.shape}')
        # resize mask to the shape of style_cond ?
        cond_mask = torch.cat([torch.from_numpy(cond_mask)]*3, dim=0).unsqueeze(0).to(opt.device)
        print(f'cond_mask.shape = {cond_mask.shape}')
        if cond_mask is not None and torch.max(cond_mask) <= 1.:
            cond_mask[cond_mask < 0.5] = (0.05 if opt.mask_ablation else 0.)
            cond_mask[cond_mask >= 0.5] = (0.95 if opt.mask_ablation else 1.)
            # TODO: check if mask smoothing is needed
        cond = cond * ( cond_mask * (0.8 if opt.mask_ablation else 1.) ) # 1 - cond_mask ?
        cv2.imwrite(ad_output, tensor2img(cond))
        adapter_features, append_to_context = get_adapter_feature(cond, adapter)
    elif opt.example_type == 'v1.5':
        adapter_features, append_to_context = None, None
        # difference between v1.5 and v1.5_adapter is just to generate adapter
        
    return {
        'sd_model': sd_model,
        'sd_sampler': sd_sampler, 
        'adapter_features': adapter_features, 
        'append_to_context': append_to_context
    }

def preload_paint_by_example_model(opt):
    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(f"{opt.example_config}")
    model = load_model_from_config(config, f"{opt.example_ckpt}").to(device)
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    return {
        'model': model,
        'sampler': sampler,
    }

def preload_sam_generator(opt):
    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.to(device=opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return {
        'sam': sam, 
        'mask_generator': mask_generator
    }

def preload_seem_detector(opt):
    cfg = load_opt_from_config_files([opt.seem_cfg])
    cfg['device'] = opt.device
    seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(opt.seem_ckpt).eval().cuda()

    return {
        'cfg': cfg,
        'sem_model': seem_model
    }
    
def preload_lama_remover(opt):

    config_path, ckpt_path = opt.lama_config, opt.lama_ckpt
    predict_config = OmegaConf.load(config_path)
    predict_config.model.path = ckpt_path
    device = torch.device(opt.device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    
    return {
        'model': model,
        'predict_config': predict_config
    }




