import torch, argparse
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'

PROMPT_BASE = ', 8K, highly detailed, expressively clear, high resolution'

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
        '--cond_inp_type',
        type=str,
        default='image',
        help='the type of the input condition image, take depth T2I as example, the input can be raw image, '
        'which depth will be calculated, or the input can be a directly a depth map image',
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
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='# of samples to generate',
    )

    return parser


def get_sd_models(opt):
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

def diffusion_inference(opt, new_target, model, sampler, **kwargs):
    # get text embedding
    # model.to(torch.float32)
    c = model.get_learned_conditioning(['a kind of  ' + new_target + PROMPT_BASE])
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
        batch_size=1,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=opt.scale,
        unconditional_conditioning=uc,
        x_T=None,
        cond_tau=opt.cond_tau,
        **kwargs
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples