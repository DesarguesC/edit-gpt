import torch, cv2, os, glob, subprocess, random
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from seem.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
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
from diffusers import DiffusionPipeline
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

def fix_mask(mask):
    dest = mask.squeeze()
    if len(dest.shape) > 2:
        dest = dest[0].unsqueeze(0).unsqueeze(0)
    else:
        dest = dest.unsqueeze(0).unsqueeze(0)
    return dest


from seem.utils.arguments import load_opt_from_config_files
from seem.modeling.BaseModel import BaseModel
from seem.modeling import build_model
from seem.utils.constants import COCO_PANOPTIC_CLASSES
from seem.demo.seem.tasks import *

seem_cfg = 'seem/configs/seem/focall_unicl_lang_demo.yaml'
seem_ckpt = '../autodl-tmp/seem_focall_v0.pt'
pldm_cfg = './configs/v1.yaml'
pldm_ckpt = '../autodl-tmp/model.ckpt'

def query_middleware(
            image: Image, 
            reftxt: str, 
            preloaded_seem_detector = None
        ):
    """
        only query mask&box for single target-noun
        
        image: removed pil image
        reftxt: query target text
    """
    if preloaded_seem_detector is None:
        cfg = load_opt_from_config_files([seem_cfg])
        cfg['device'] = 'cuda'
        seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(seem_ckpt).eval().cuda()
    else:
        cfg = preloaded_seem_detector['cfg']
        seem_model = preloaded_seem_detector['seem_model']

    with torch.no_grad():
        seem_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"],
                                                                                 is_eval=True)
    # get text-image mask
    width, height = image.size
    image_ori = np.asarray(image)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    data = {"image": images, "height": height, "width": width}
    visual = Visualizer(image_ori, metadata=metadata)

    seem_model.model.task_switch['spatial'] = False
    seem_model.model.task_switch['visual'] = False
    seem_model.model.task_switch['grounding'] = False
    seem_model.model.task_switch['audio'] = False
    seem_model.model.task_switch['grounding'] = True

    data['text'] = [reftxt]
    batch_inputs = [data]

    results, image_size, extra = seem_model.model.evaluate_demo(batch_inputs)
    # print(f'extra.keys() = {extra.keys()}')
    # print(f'results.keys() = {results.keys()}')

    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = seem_model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id,:,:]
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    # mask queried from text
    pred_box_pos = None
    demo = visual.draw_binary_mask(pred_masks_pos.squeeze(), text=reftxt)  # rgb Image
    res = demo.get_image()

    # TODO: save
    sam_output_dir = os.path.join('../Semantic')
    if not os.path.exists(sam_output_dir): os.mkdir(sam_output_dir)
    name_ = f'./{sam_output_dir}/panoptic'
    t = 0
    while os.path.isfile(f'{name_}-{t}.jpg'): t += 1
    cv2.imwrite(f'{name_}-{t}.jpg', cv2.cvtColor(np.uint8(res), cv2.COLOR_RGB2BGR))
    print(f'seg result image saved at \'{name_}-{t}.jpg\'')


    return Image.fromarray(res), pred_masks_pos, pred_box_pos


# dog = Image.open('../dog.jpg')
# prompt = "a sad dog"

# _, mask, _ = query_middleware(dog, prompt)
# print(f'mask.shape = {mask.shape}')

# from einops import rearrange, repeat
# from basicsr.utils import img2tensor, tensor2img


# mask = rearrange(repeat(mask, '1 h w -> c h w', c=3), 'c h w -> h w c')
# if np.max(mask) <= 1.: mask = mask * 255.
# mask = Image.fromarray(np.uint8(mask)).convert('RGB').save('../dog-mask.jpg')
from einops import rearrange, repeat

base_img = Image.open('./assets/tie.jpg')
ref = Image.open('../dog.jpg')
mask = rearrange(torch.tensor(np.array(ImageOps.fit(Image.open('../dog-mask.jpg').convert('RGB'), (448//8, 576//8),method=Image.Resampling.LANCZOS))), 'h w c -> c h w')[0]




mask = mask.unsqueeze(0).unsqueeze(0)
print(f'mask.shape = {mask.shape}')

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

def paint_by_example(
        mask: torch.Tensor = None,
        ref_img: Image = None,
        base_img: Image = None,
        preloaded_example_painter=None,
        **kwargs
):
    # mask: [1, 1, h, w] is required
    # assert ref_img.size == base_img.size, f'ref_img.size = {ref_img.size}, base_img.size = {base_img.size}'
    mask = fix_mask(mask)  # fix dimensions
    print(f'Example Mask = {mask.shape}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if preloaded_example_painter is None:
        seed_everything(42)
        config = OmegaConf.load(pldm_cfg)
        model = load_model_from_config(config, pldm_ckpt).to(device)
        # if opt.plms:
        sampler = PLMSSampler(model)
        # else:
        #     sampler = DDIMSampler(model)
    else:
        model = preloaded_example_painter['model']
        sampler = preloaded_example_painter['sampler']

    op_output = '../dog-out.jpg'
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                image_tensor = get_tensor()(base_img.convert('RGB')).unsqueeze(0)
                mask[mask < 0.5] = 0.
                mask[mask >= 0.5] = 1.
                mask_tensor = 1. - mask.to(torch.float32)
                
                ref_p = ref_img.convert('RGB').resize((224, 224))
                ref_tensor = get_tensor_clip()(ref_p).unsqueeze(0).to(device)

                print(
                    f'image_tensor.shape = {image_tensor.shape}, mask_tensor.shape = {mask.shape}, ref_tensor.shape = {ref_tensor.shape}')
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
                c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                c = model.proj_out(c)

                # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                shape = [4, 576 // 8, 448 // 8]

                samples_ddim, _ = sampler.sample(
                    S=50,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=7.5,
                    unconditional_conditioning=uc,
                    x_T=None,
                    adpater_features=None,
                    append_to_context=None,
                    test_model_kwargs=test_model_kwargs,
                    **kwargs
                )
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.) / 2., min=0., max=1.)

    return op_output, x_samples_ddim


path, x_sample = paint_by_example(mask, ref, base_img)

Image.fromarray(cv2.cvtColor(np.uint8(tensor2img(x_sample)), cv2.COLOR_RGB2BGR)).save(path)
