import torch, argparse, cv2
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from enum import Enum, unique

from basicsr.utils import img2tensor
from torch import autocast

from ldm.modules.encoders.adapter import Adapter, StyleAdapter, Adapter_light
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict, resize_numpy_image
from prompt.guide import get_response, first_ask_expand

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'\
                          'anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured'

PROMPT_BASE = '8K, highly detailed, expressively clear, high resolution'

from prompt.guide import Use_Agent

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_base_argument_parser(parser) -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    # parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir',
        type=str,
        help='dir to write results to',
        default=None,
    )

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='positive prompt',
    )

    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )

    parser.add_argument(
        '--cond_path',
        type=str,
        default=None,
        help='condition image path',
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default='ddim',
        choices=['ddim', 'plms'],
        help='sampling algorithm, currently, only ddim and plms are supported, more are on the way',
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of sampling steps',
    )

    parser.add_argument(
        '--sd_ckpt',
        type=str,
        default='../autodl-tmp/v1-5-pruned.ckpt',
        help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
    )

    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default=None,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
    )

    parser.add_argument(
        '--sd_config',
        type=str,
        default='./configs/sd-v1-inference.yaml',
        help='path to config which constructs SD model',
    )

    parser.add_argument(
        '--max_resolution',
        type=float,
        default=512 * 512,
        help='max image height * width, only for computer with limited vram',
    )

    parser.add_argument(
        '--resize_short_edge',
        type=int,
        default=None,
        help='resize short edge of the input image, if this arg is set, max_resolution will not be used',
    )

    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )

    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor',
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )

    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau')

    parser.add_argument(
        '--cond_weight',
        type=float,
        default=1.0,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        '--txt_cfg',
        type=float,
        default=7.5,
        help='classifier-free guidance of text in InstructPix2Pix'
    )

    parser.add_argument(
        '--img_cfg',
        type=float,
        default=1.5,
        help='classifier-free guidance of image in InstructPix2Pix'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='# of samples to generate',
    )
    
    parser.add_argument(
        '--ddim_eta',
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)"
    )
    
    parser.add_argument(
        '--fixed_code',
        action='store_true',
        help="if enabled, uses the same starting code across samples"
    )
    
    parser.add_argument(
        '--precision',
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument(
        '--linear',
        type=str2bool,
        help="user linearart adpater or depth adapter in XL",
        default=True   # default: linear
    )
    
    parser.add_argument(
        '--use_lama',
        type=str2bool,
        help="whether to use LaMa for remove inpainting",
        default=True
    )
    
    parser.add_argument(
        '--dilate_kernel_size',
        type=int,
        help="kernel size to deliate when inpainting via LaMa",
        default=15
    )
    
    parser.add_argument(
        '--mask_ablation',
        type=str2bool,
        help="whether to do the mask ablation",
        default=False
    )

    parser.add_argument(
        '--preload_all_models',
        type=str2bool,
        default=False,
        help='preload all models for valuation',
    )

    parser.add_argument(
        '--preload_all_agents',
        type=str2bool,
        default=False,
        help='preload all agents for valuation',
    )

    parser.add_argument(
        '--use_max_min',
        type=str2bool,
        default=False,
        help='For SAM ablation, whether to use max-min method to create mask or <SAM,SEEM>',
    )

    parser.add_argument(
        '--gpt4_v',
        type=str2bool,
        default=False,
        help='ablation for <SEEM+GPT3.5> or <IMG+GPT4V>',
    )

    parser.add_argument(
        '--compile',
        type=str2bool,
        default=False,
        help='when you are testing system on a huge system, you should set it as True'
    )

    parser.add_argument(
        '--with_ip2p_val',
        type=str2bool,
        default=False,
        help='whether to valuate ip2p simultaneously'
    )

    return parser

def get_sd_models(opt):
    
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.models.diffusion.plms import PLMSSampler
    """
    build stable diffusion model, sampler
    """
    # SD
    config = OmegaConf.load(f"{opt.sd_config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    sd_model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return sd_model, sampler



def diffusion_inference(opt, prompts, model, sampler, adapter_features=None, append_to_context=None, **kwargs):
    # get text embedding
    c = model.get_learned_conditioning([prompts])
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning([opt.neg_prompt])
    else:
        uc = None
    c, uc = fix_cond_shapes(model, c, uc)
    

    if not hasattr(opt, 'H'):
        opt.H = 512
        opt.W = 512
    
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    samples_latents, _ = sampler.sample(
        S=opt.steps,
        conditioning=c,
        batch_size=opt.n_samples,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=opt.scale,
        unconditional_conditioning=uc,
        x_T=kwargs['start_code'] if 'start_code' in kwargs.keys() else None,
        eta=opt.ddim_eta,
        adapter_features=adapter_features,
        append_to_context=append_to_context,
        cond_tau=opt.cond_tau
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples









