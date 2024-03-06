from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.lineart import LineartDetector
import torch
from PIL import Image

# load adapter
adapter = T2IAdapter.from_pretrained(
  "../autodl-tmp/TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16, varient="fp16", local_files_only=True
).to("cuda")

# load euler_a scheduler
model_id = '../autodl-tmp/stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True)
vae=AutoencoderKL.from_pretrained("../autodl-tmp/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, local_files_only=True)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", local_files_only=True 
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

line_detector = LineartDetector.from_pretrained("../autodl-tmp/lllyasviel/Annotators").to("cuda")

url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_lin.jpg"
# image = load_image(url)
image = Image.open('./assets/dog.jpg').resize((2048,1024))
print(type(image), image.size)
image = line_detector(
    image, detect_resolution=384, image_resolution=1024
)
print(image.size)
image.save('./outputs/detect-384.png')

prompt = "Ice dragon roar, 4k photo"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
gen_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    num_inference_steps=30,
    adapter_conditioning_scale=0.8,
    guidance_scale=7.5, 
).images[0]
print(gen_images.size)

gen_images.save('./outputs/out_lin.png')

