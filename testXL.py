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
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline


# from diffusers import DiffusionPipeline
# import torch

# pipe = DiffusionPipeline.from_pretrained("../autodl-tmp/stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe.to("cuda")

# # if using torch < 2.0
# # pipe.enable_xformers_memory_efficient_attention()

# prompt = "An astronaut riding a green horse"

# images = pipe(prompt=prompt).images[0]
# images.save('./testXL.jpg')
# exit(0)


pipe = DiffusionPipeline.from_pretrained(f"../autodl-tmp/stabilityai/stable-diffusion-xl-base-1.0", \
                                                    torch_dtype=torch.float16, use_safetensors=True, variant="fp16"  )
pipe.to("cuda")
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    f"../autodl-tmp/stabilityai/stable-diffusion-xl-refiner-1.0",
                    text_encoder_2 = pipe.text_encoder_2,
                    vae = pipe.vae,
                    torch_dtype = torch.float16, 
                    use_safetensors = True, 
                    variant = "fp16", 
                )
refiner.to('cuda')

prompts = 'a beautiful bird'

steps = 40


gen_images = pipe(
    prompt = prompts,
    num_inference_steps = steps,
    guidance_scale = 7.5,
    # output_type = 'latent',
).images[0]
gen_images.save('test-xl-1.jpg')


gen_images = refiner(
    prompt = prompts,
    num_inference_steps = steps,
    denoising_end = 1,
    image = gen_images,
).images[0]

gen_images.save(f'test-xl-2.jpg')

# # load adapter
# adapter = T2IAdapter.from_pretrained(
#   "../autodl-tmp/TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16, varient="fp16", local_files_only=True
# ).to("cuda")

# # load euler_a scheduler
# model_id = '../autodl-tmp/stabilityai/stable-diffusion-xl-base-1.0'
# euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True)
# vae=AutoencoderKL.from_pretrained("../autodl-tmp/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, local_files_only=True)
# pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
#     model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", local_files_only=True 
# ).to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

# line_detector = LineartDetector.from_pretrained("../autodl-tmp/lllyasviel/Annotators").to("cuda")

# url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_lin.jpg"
# # image = load_image(url)
# image = Image.open('./assets/dog.jpg').resize((2048,1024))
# print(type(image), image.size)
# image = line_detector(
#     image, detect_resolution=384, image_resolution=1024
# )
# print(image.size)
# image.save('./outputs/detect-384.png')

# prompt = "Ice dragon roar, 4k photo"
# negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
# gen_images = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     image=image,
#     num_inference_steps=30,
#     adapter_conditioning_scale=0.8,
#     guidance_scale=7.5, 
# ).images[0]
# print(gen_images.size)

# gen_images.save('./outputs/out_lin.png')

