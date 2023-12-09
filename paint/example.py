
import argparse, os, sys, glob
import cv2, torch, time, clip
import numpy as np
# from basicsr.utils import tensor2img
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
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from torchvision.transforms import Resize

wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

from ldm.util import load_model_from_config
from ldm.inference_base import diffusion_inference



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

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

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

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



def paint_by_example(opt, mask: torch.Tensor = None, ref_img: Image = None, base_img: Image = None):
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

    image_tensor = get_tensor()(base_img.conver('RGB')).unsqueeze(0)

    mask[mask < 0.5] = 0.
    mask[mask >= 0.5] = 1.
    mask_tensor = mask.to(torch.float32)
    assert mask_tensor.shape[-2:] == (opt.H, opt.W), f'mask_tensor.shape = {mask_tensor.shape}'

    ref_p = ref_img.convert('RGB').resize((opt.W,opt.H))
    ref_tensor = get_tensor_clip()(ref_p).unsqueeze(0).to(device)

    print(f'image_tensor.shape = {image_tensor.shape}, mask_tensor.shape = {mask.shape}')
    inpaint_image = image_tensor * mask_tensor

    test_model_kwargs = {
        'inpaint_mask': mask_tensor.to(device),
        'inpaint_image': inpaint_image.to(device)
    }

    uc = None
    if opt.scale != 1.0:
        uc = model.learnable_vector
    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
    c = model.proj_out(c)

    inpaint_mask = test_model_kwargs['inpaint_mask']
    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
    test_model_kwargs['inpaint_image'] = z_inpaint
    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
        test_model_kwargs['inpaint_mask'])
    
    uc = None
    if opt.scale != 1.0:
        uc = model.learnable_vector
    c = model.get_learned_conditioning(new_target.to(torch.float16))
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
        test_model_kwargs=test_model_kwargs,
    )
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.) / 2., min=0., max=1.)
    
    # diffusion_inference(opt, new_target=ref_tensor, model=model,
                            # sampler=sampler, start_code=start_code, test_model_kwargs=test_model_kwargs)
    # cv2.imwrite(output_path, tensor2img(x_samples_ddim))
    return output_path, x_samples_ddim

    # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    # _ = check_safety(x_samples_ddim)
    # x_checked_image = x_samples_ddim
    # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)



