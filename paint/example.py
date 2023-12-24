import argparse, os, sys, glob
import cv2, torch, time, clip
import numpy as np
from basicsr.utils import tensor2img
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from pldm.util import instantiate_from_config, load_model_from_config
from pldm.models.diffusion.ddim import DDIMSampler
from pldm.models.diffusion.plms import PLMSSampler
from paint.control import *
from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize
from paint.control import get_adapter, get_adapter_feature, get_style_model, process_style_cond
from ldm.inference_base import *
from prompt.guide import *


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def generate_example(opt, new_noun, expand_agent=None, ori_img: Image = None, cond_mask=None, **kwargs) -> Image:
    
    opt.XL_base_path = opt.XL_base_path.strip('/')
    
    # expand_agent=None
    
    prompts = f'a photo of ONLY a/an {new_noun}, {PROMPT_BASE}'
    prompts = f'{prompts}. Simultaneously, {get_response(expand_agent, new_noun)}' if expand_agent != None else prompts
    print(f'prompt: \n {prompts}\n')
    if opt.example_type == 'XL_adapter':
        print('-' * 9 + 'Generating via SDXL-Adapter' + '-' * 9)
        from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
        from controlnet_aux.lineart import LineartDetector
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
        line_detector = LineartDetector.from_pretrained(f"{opt.XL_base_path}/lllyasviel/Annotators").to("cuda")

        image = line_detector(
            ori_img, detect_resolution=384, image_resolution=opt.H
        )
        if cond_mask is not None:
            if np.abs(np.max(cond_mask)-1.) < 0.01:
                cond_mask[cond_mask < 0.5] = 0.
                cond_mask[cond_mask >= 0.5] = 1.
            cond_mask = repeat(rearrange(cond_mask, 'c h w -> h w c'), '... 1 -> ... c', c=3)
            image = image * cond_mask
            image = Image.fromarray(np.uint8(image * cond_mask))
        
        gen_images = pipe(
            prompt=prompts,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            image=image,
            num_inference_steps=opt.steps,
            adapter_conditioning_scale=0.8,
            guidance_scale=7.5,
        ).images[0]

    elif opt.example_type == 'XL':
        print('-' * 9 + 'Generating via SDXL-Base' + '-' * 9)
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(f"{opt.XL_base_path}/stabilityai/stable-diffusion-xl-base-1.0", \
                                                 torch_dtype=torch.float16, use_safetensors=True, variant="fp16", local_files_only=True)
        pipe.to("cuda")
        gen_images = pipe(prompts, height=opt.H, width=opt.W, steps=int(1.5*opt.steps)).images[0]
    
    else:  # 'v1.5' in opt.example_type
        print('-'*9 + 'Generating via sd1.5 with T2I-Adapter' + '-'*9)
        sd_model, sd_sampler = get_sd_models(opt)
        # ori_img: for depth condition generating
        adapter_features, append_to_context = None, None

        if opt.example_type == 'v1.5_adapter':
            print('-'*9 + 'Generating via Style Adapter' + '-'*9)
            adapter, cond_model = get_adapter(opt, cond_type='depth'), get_depth_model(opt)
            print(f'BEFORE: cond_img.size = {ori_img.size}')
            cond = process_depth_cond(opt, ori_img, cond_model) # not a image
            print(f'cond.shape = {cond.shape}, cond_mask.shape = {cond_mask.shape}')
            # resize mask to the shape of style_cond ?
            cond_mask = torch.cat([torch.from_numpy(cond_mask)]*3, dim=0).unsqueeze(0).to(opt.device)
            print(f'cond_mask.shape = {cond_mask.shape}')
            if cond_mask is not None:
                cond_mask[cond_mask < 0.5] = 0.05 if opt.mask_ablation else 0.
                cond_mask[cond_mask >= 0.5] = 0.95 if opt.mask_ablation else 1.
                # TODO: check if mask smoothing is needed
            cond = cond * (cond_mask * 0.8 if opt.mask_ablation else 1.) # 1 - cond_mask ?
            cv2.imwrite(f'./static/adapter_cond.jpg', tensor2img(cond))
            adapter_features, append_to_context = get_adapter_feature(cond, adapter)
        
        # if isinstance(adapter_features, list):
        #     print(len(adapter_features))
        # else:
        #     print(adapter_features)
        # print(append_to_context)

        with torch.inference_mode(), sd_model.ema_scope(), autocast('cuda'):
            seed_everything(opt.seed)
            diffusion_image = diffusion_inference(opt, prompts, sd_model, sd_sampler, adapter_features=adapter_features, append_to_context=append_to_context)
            diffusion_image = tensor2img(diffusion_image)
            cv2.imwrite('./static/ref.jpg', diffusion_image)
            print(f'example saved at \'./static/ref.jpg\'')
            diffusion_image = cv2.cvtColor(diffusion_image, cv2.COLOR_BGR2RGB)
        del sd_model, sd_sampler
        gen_images = Image.fromarray(np.uint8(diffusion_image)).convert('RGB') # pil
    
    
    if gen_images.size != ori_img.size:
        print(f'gen_images.size = {gen_images.size}, ori_img.size = {ori_img.size}')
        gen_images = gen_images.resize(ori_img.size)
    assert gen_images.size == ori_img.size, f'gen_images.size = {gen_images.size}, ori_img.size = {ori_img.size}'
    
    gen_images.save(f'./static/ref4{opt.out_name}')
    print(f'example saved at \'./static/ref4{opt.out_name}\' [sized: {gen_images.size}]')
    return gen_images # PIL.Image


def paint_by_example(opt, mask: torch.Tensor = None, ref_img: Image = None, base_img: Image = None, use_adapter=False, style_mask=None, **kwargs):
    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load(f"{opt.example_config}")
    model = load_model_from_config(config, f"{opt.example_ckpt}").to(device)
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    output_path = os.path.join(opt.out_dir, opt.out_name)
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    adapter_features = append_to_context = None
    if use_adapter and style_mask is not None:
        print('-' * 9 + 'Example created via Adapter' + '-' * 9)
        adapter, cond_model = get_adapter(opt), get_style_model(opt)
        style_cond = process_style_cond(opt, ref_img, cond_model)
        assert style_mask is not None
        print(f'style_cond.shape = {style_cond.shape}, style_mask.shape = {style_mask.shape}')
        if style_mask is not None:
            style_mask[style_mask < 0.5] = 0.
            style_mask[style_mask >= 0.5] = 1.
            # TODO: check if mask smoothing is needed
            style_cond *= style_mask
        cv2.imwrite(f'./static/style_cond.jpg', tensor2img(style_cond))
        adapter_features, append_to_context = get_adapter_feature(style_cond, adapter)
        # adapter_features got!


    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                
                image_tensor = get_tensor()(base_img.convert('RGB')).unsqueeze(0)
                mask[mask < 0.5] = 0.
                mask[mask >= 0.5] = 1.
                mask_tensor = 1. - mask.to(torch.float32)
                assert mask_tensor.shape[-2:] == (opt.H, opt.W), f'mask_tensor.shape = {mask_tensor.shape}'

                ref_p = ref_img.convert('RGB').resize((224,224))
                ref_tensor = get_tensor_clip()(ref_p).unsqueeze(0).to(device)

                print(f'image_tensor.shape = {image_tensor.shape}, mask_tensor.shape = {mask.shape}, ref_tensor.shape = {ref_tensor.shape}')
                inpaint_image = image_tensor * mask_tensor

                test_model_kwargs = {
                    'inpaint_mask': mask_tensor.to(device),
                    'inpaint_image': inpaint_image.to(device)
                }

                z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                test_model_kwargs['inpaint_image'] = z_inpaint
                test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                                                                                test_model_kwargs['inpaint_mask'])

                uc = None
                if opt.scale != 1.0:
                    uc = model.learnable_vector
                c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                c = model.proj_out(c)

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                samples_ddim, _ = sampler.sample(
                    S=opt.steps,
                    conditioning=c,
                    batch_size=opt.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    x_T=start_code,
                    adpater_features=adapter_features,
                    append_to_context=append_to_context,
                    test_model_kwargs=test_model_kwargs,
                    **kwargs
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.) / 2., min=0., max=1.)
    
    return output_path, x_samples_ddim




