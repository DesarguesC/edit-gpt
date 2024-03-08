from typing import Tuple
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import os
import numpy as np
from omegaconf import OmegaConf
import importlib
from ldm.util import instantiate_from_config
from torch.nn import functional as F
from einops import repeat, rearrange
from paint.control import *

to_tensor = ToTensor()

inpaint_config_path = './configs/latent-diffusion/gqa-inpaint-ldm-vq-f8-256x256.yaml'
inpaint_model_base_path = './inst-paint'

def load_inpaint_model(
    ckpt_base_path = inpaint_model_base_path, config_path = inpaint_config_path, device='cuda'
    ):
    parsed_config = OmegaConf.load(config_path)
    ckpt = os.path.join(ckpt_base_path, 'ldm/model.ckpt')
    print(f'Loading model from: {ckpt}')
    # sd = read_state_dict(ckpt)
    model = instantiate_from_config(parsed_config["model"])
    
    model_state_dict = torch.load(ckpt, map_location=device)["state_dict"]
    
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    
    return model

def preprocess_image(
        image: Image, resize_shape: Tuple[int, int] = (256, 256), center_crop=False
    ):
    pil_image = image

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        pil_image = image.crop((left, top, right, bottom))

    pil_image = pil_image.resize(resize_shape)

    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0) * 2 - 1

    return pil_image, tensor_image

def target_removing(
        opt, target_noun: str, image: Image, model=None, resize_shape: Tuple[int, int] = (256, 256),
        ori_shape: Tuple[int, int] = (512, 512), recovery=True, center_crop=False, mask=None
    ) -> Image:
    # print(f'start => image.size = {image.size}')
    # print(f'start => mask.shape = {mask.shape}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ori_shape => image.shape
    model = load_inpaint_model(ckpt_base_path=opt.inpaint_folder, config_path=opt.inpaint_config, device=device) if model==None else model
    pil_image_pointer = image
    
    # if opt.use_inpaint_adapter:
    #     # depth

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2
        pil_image_pointer = image.crop((left, top, right, bottom))
        # if used, connect it with the uncopped part

    pil_image = pil_image_pointer.resize(resize_shape)
    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0) * 2 - 1
    print(f'tensor_image.shape = {tensor_image.shape}')

    rmtxt = 'remove the ' + target_noun
    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32, requires_grad=False)
        mask = repeat(mask, '1 ... -> c ...', c=4)[None].to(device)
        # print(f'mask.shape = {mask.shape}')
        h, w = resize_shape
        mask = F.interpolate(
            mask,
            size=(h//opt.f, w//opt.f),
            mode='bilinear',
            align_corners=False
        )
        mask[mask < 0.5] = 0.
        mask[mask >= 0.5] = 1.
    
    pil_removed = model.inpaint(tensor_image, rmtxt, num_steps=50, device=device, return_pil=True, seed=0, mask=mask if mask is not None else None)
    if recovery: pil_removed = pil_removed.resize(ori_shape)
    return pil_removed

def match_sam_box(mask: np.array, sam_list: list[tuple]):
    pointer = sam_list
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    box_idx = np.argmax([
        np.sum(mask.squeeze() * sam_[1].squeeze()) / np.sum(mask) for sam_ in pointer
    ])
    bbox = sam_list[box_idx][0]
    del pointer[box_idx]
    return bbox

def refactor_mask(box_1, mask_1, box_2, type='remove'):
    """
        mask_1 is in box_1
        reshape mask_1 into box_2, as mask_2, return
        TODO: refactor mask_1 into box_2 (tend to get smaller ?)
    """
    mask_1 = torch.tensor(mask_1.squeeze(), dtype=torch.float32)
    # h * w
    mask_2 = torch.zeros_like(mask_1.unsqueeze(0))
    # 1 * h * w
    print(f'box_1 = {box_1}, mask_1.shape = {mask_1.shape}, box_2 = {box_2}, mask_2.shape = {mask_2.shape}')
    x1, y1, w1, h1 = box_1
    x2, y2, w2, h2 = box_2
    x1, x2, y1, y2, w1, w2, h1, h2 = int(x1), int(x2), int(y1), int(y2), int(w1), int(w2), int(h1), int(h2)
    print(f'x1 = {x1}, y1 = {y1}, w1 = {w1}, h1 = {h1}')
    print(f'x2 = {x2}, y2 = {y2}, w2 = {w2}, h2 = {h2}')
    valid_mask = mask_1.unsqueeze(0)[:, y1:y1+h1,x1:x1+w1]
    valid_mask = rearrange(valid_mask, 'c h w -> 1 c h w')
    # print(f'valid_mask.shape = {valid_mask.shape}')
    resized_valid_mask = F.interpolate(
        valid_mask,
        size=(h2, w2),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    # resized_valid_mask = rearrange(repeat(rearrange(resized_valid_mask, 'h w -> 1 h w'), '1 h w -> c h w', c=3), 'c h w -> 1 c h w')
    resized_valid_mask = resized_valid_mask.unsqueeze(0)
    resized_valid_mask[resized_valid_mask > 0.5] = 1.
    resized_valid_mask[resized_valid_mask <= 0.5] = 0.
    print(f'resized_valid_mask.shape = {resized_valid_mask.shape}')
    # 1 * h * w
    print(f'part: mask_2[:, y2:y2+h2, x2:x2+w2].shape = {mask_2[:, y2:y2+h2, x2:x2+w2].shape}')
    # 1 * w * h
    mask_2[:,y2:y2+h2,x2:x2+w2] = resized_valid_mask
    assert mask_2.shape == mask_1.shape, f'mask_1.shape = {mask_1.shape}, mask_2.shape = {mask_2.shape}'
    return repeat(mask_2, '1 h w -> c h w', c = 3).unsqueeze(0) if type == 'remove' else mask_2
    # 1 * 3 * h * w

# TODO: implement 'refact_target'

def fix_box(gpt_box, img_shape):
    assert len(gpt_box) == 4 and len(img_shape) == 3
    x, y, w, h = gpt_box
    fixed_box = (0,0,0,0)
    if w < 0: 
        w = -w
        print('?')
    if h < 0: 
        h = -h
        print('?')
    H_, W_, _ = img_shape # [h, w, 3]
    flag = 0
    while x + w >= W_ or y + h >= H_:
        w //= 2
        h //= 2
        flag += 1
        if flag > 4:
            return fixed_box
    fixed_box = (x,y,w,h)
    return fixed_box
