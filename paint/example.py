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

from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize
from paint.control import get_adapter, get_adapter_feature, get_style_model, process_style_cond


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


def paint_by_example(opt, mask: torch.Tensor = None, ref_img: Image = None, base_img: Image = None, use_adapter=False, style_mask=None):
    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load(f"{opt.example_config}")
    model = load_model_from_config(config, f"{opt.example_ckpt}").to(device)
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    output_path = os.path.join(opt.out_dir, opt.out_name)
    batch_size = opt.n_samples # use default value: 1
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    adapter_features = append_to_context = None
    if use_adapter:
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
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.) / 2., min=0., max=1.)
    
    return output_path, x_samples_ddim




