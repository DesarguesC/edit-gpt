from omegaconf import OmegaConf
import k_diffusion as K
from PIL import Image
from einops import repeat, rearrange
import numpy as np
import torch, cv2, os, random, math, yaml
from torch import nn, autocast

from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything
from ldm.util import load_model_from_config
from ldm.inference_base import *
from paint.control import get_adapter, get_adapter_feature, get_style_model, process_style_cond
from paint.control import *
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry

from seem.utils.arguments import load_opt_from_config_files
from seem.modeling.BaseModel import BaseModel
from seem.modeling import build_model
from seem.utils.constants import COCO_PANOPTIC_CLASSES
from seem.demo.seem.tasks import *
from seem.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL, DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from controlnet_aux.lineart import LineartDetector
from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize

from lama.saicinpainting.training.trainers import load_checkpoint

A = 'find target to be removed'
B = 'find target to be replaced'
C = 'rescale bbox for me'
D = 'expand diffusion prompts for me'
E = 'arrange a new bbox for me'
F = 'find target to be moved'
G = 'find target to be added'
H = 'generate a new bbox for me'
I = 'adjust bbox for me'
V = 'use gpt-4v'


def preload_ip2p(opt):
    seed_everything(opt.seed)
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
    seed_everything(opt.seed)
    pipe = DiffusionPipeline.from_pretrained(f"{opt.XL_base_path}/stabilityai/stable-diffusion-xl-base-1.0", \
                                                torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    f"{opt.XL_base_path}/stabilityai/stable-diffusion-xl-refiner-1.0",
                    text_encoder_2 = pipe.text_encoder_2,
                    vae = pipe.vae,
                    torch_dtype = torch.float16, 
                    use_safetensors = True, 
                    variant = "fp16", 
                )
    refiner.to('cuda')                        

    return {
        'pipe': pipe,
        'refiner': refiner
    }

def preload_XL_adapter_generator(opt):
    seed_everything(opt.seed)
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

def preload_v1_5_generator(opt):
    # for both v1.5 and v1.5_adapter
    seed_everything(opt.seed)
    sd_model, sd_sampler = get_sd_models(opt)
    if opt.example_type == 'v1.5_adapter':
        print('-'*9 + 'Generating via Style Adapter (depth)' + '-'*9)
        adapter, cond_model = get_adapter(opt, cond_type='depth'), get_depth_model(opt)
    else:
        adapter, cond_model = None, None
        adapter_features, append_to_context = None, None
        # opt.example_type == 'v1.5'
        # difference between v1.5 and v1.5_adapter is just to generate adapter


    return {
        'sd_model': sd_model,
        'sd_sampler': sd_sampler,
        'adapter': adapter,
        'cond_model': cond_model,
        'adapter_features': None,
        'append_to_context': None
    }

def preload_example_generator(opt):
    seed_everything(opt.seed)
    if opt.example_type == 'XL':
        return preload_XL_generator(opt)
    elif opt.example_type == 'XL_adapter':
        return preload_XL_adapter_generator(opt)
    else:  # stable-diffusion 1.5
        return preload_v1_5_generator(opt)

def preload_paint_by_example_model(opt):
    from pldm.models.diffusion.ddim import DDIMSampler
    from pldm.models.diffusion.plms import PLMSSampler
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
    seed_everything(opt.seed)
    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.to(device=opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return {
        'sam': sam, 
        'mask_generator': mask_generator
    }

def preload_seem_detector(opt):
    seed_everything(opt.seed)
    cfg = load_opt_from_config_files([opt.seem_cfg])
    cfg['device'] = opt.device
    seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(opt.seem_ckpt).eval().cuda()

    return {
        'cfg': cfg,
        'seem_model': seem_model
    }
    
def preload_lama_remover(opt):
    seed_everything(opt.seed)
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

def preload_refiner(opt):
    return  StableDiffusionXLImg2ImgPipeline.from_pretrained(
        f"{opt.XL_base_path}/stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True
    ).to(opt.device)



def preload_all_models(opt):
    seed_everything(opt.seed)
    preloaded_model =  {
        'preloaded_ip2p': preload_ip2p(opt), # 8854 MiB
        'preloaded_example_generator': preload_example_generator(opt), 
        # XL - 8272 MiB, XL_ad - 8458 MiB, V1.5 - 10446 MiB
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt) # 10446 MiB
    }
    if opt.example_type != 'XL':
        preloaded_model['preloaded_refiner'] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                f"{opt.XL_base_path}/stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(opt.device)
    return preloaded_model


def preload_all_agents(opt):
    # task planning agent need to be added outside
    return {
        A: Use_Agent(opt, TODO = A),
        B: Use_Agent(opt, TODO = B),
        C: Use_Agent(opt, TODO = C),
        D: Use_Agent(opt, TODO = D),
        E: Use_Agent(opt, TODO = E),
        F: Use_Agent(opt, TODO = F),
        G: Use_Agent(opt, TODO = G),
        H: Use_Agent(opt, TODO = H),
        I: Use_Agent(opt, TODO = I),
    }





