from typing import Tuple

import cv2
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
from random import randint
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

def max_min_box(mask0):
    mask0[mask0>0.5] = 1
    mask0[mask0<=0.5] = 0
    # print(f'TEST-min-max: mask.shape = {mask0.shape}') # 1 * h * w
    H, W = mask0.shape
    max_x, max_y = 0, 0
    min_x, min_y = W - 1, H - 1

    for i in range(H):
        for j in range(W):
            if mask0[i][j].item() == 1:
            # if (len(mask0.shape) == 2 and mask0[i][j].item() == 1) or (len(mask0.shape) == 3 and mask0[0][i][j].item() == 1) or (len(mask0.shape) == 4 and mask0[0][0][i][j].item() == 1):
                max_x, min_x = max(j, max_x), min(j, min_x)
                max_y, min_y = max(i, max_y), min(i, min_y)
    return (min_x, min_y, max_x-min_x, max_y-min_y)


def match_sam_box(mask: np.array = None, sam_list: list[tuple] = None, use_max_min=False, use_dilation=False, dilation=1, dilation_iter=4):
    # not deal with ratio mode yet, return normal integer box elements
    # dilation: opt.erosion, dilation_iter_num: opt.erosion_iter_num
    assert mask is not None, f'mask is None'
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    # Senseless: use_dilation, but not use_max_min
    if use_dilation and use_max_min:
        # print(f'[Before Dilation] mask.shape = {mask.shape} | np.max(mask) = {np.max(mask)}')
        # Add: [h, w], Move: [1, h, w] ?
        while len(mask.shape) > 2 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0) # ?
        # convert to [h, w]
        if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
        if np.max(mask) <= 1.: mask = mask * 255
        mask = np.uint8(mask)

        # use opencv dilation instead of SAM/max_min
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
        eroded = cv2.erode(mask, kernel, iterations=dilation_iter) # mask: [h, w]
        # set (erosion, erosion_iter_num) = (1,x), (2,1), (3,1) are better (erosion kernel > 3 becomes useless)
        """
            not using np.squeeze: msak [1 h w] seems to lead to an error ?
        """
        print(f'eroded.shape = {eroded.shape}')
        return max_min_box(eroded)
    elif use_max_min:
        return max_min_box(mask)    # use max & min coordinates for bounding box generating

    pointer = sam_list
    box_idx = np.argmax([
        np.sum(mask.squeeze() * sam_[1].squeeze()) / np.sum(mask) for sam_ in pointer
    ])
    bbox = sam_list[box_idx][0]
    del pointer[box_idx]
    x, y, w, h = bbox
    
    return (int(x), int(y), int(w), int(h))

def move_ref2base(box, ref_img, ori_img, mask: torch.Tensor = None):
    """
        put ref_img, multiplied by mask, put to ref to the ori_img[box2]
    """
    # mask: [1, 3, h, w]
    mask = np.array(rearrange(mask.squeeze(), 'c h w -> h w c'))
    if isinstance(ref_img, Image.Image):
        ref_img = np.array(ref_img) # [h w 3]
    x, y, w, h = box
    ref_img = cv2.resize(ref_img, (w,h))
    if isinstance(ori_img, Image.Image):
        ori_img = np.array(ori_img)
    need_ori = (1 - mask) * ori_img
    ori_img[y:y+h, x:x+w, :] = ref_img # directional replacement
    ori_img = mask * ori_img + need_ori
    return Image.fromarray(np.uint8(ori_img), mode='RGB'), np.uint8(ori_img), mask




def refactor_mask(box_1, mask_1, box_2, type='remove'):
    """
        mask_1: [1, h, w]
        box_1 is in mask_1
        reshape mask_1[box_1] to box_2, and add it to the zero matrix mask_2
        reshape mask_1 into box_2, as mask_2, return
    """
    # for ablation study, calculate box_1 (corresponding to mask_1)  via max-min coordinates

    mask_1 = torch.tensor(mask_1.squeeze(), dtype=torch.float32) # h * w
    mask_2 = torch.zeros_like(mask_1.unsqueeze(0)) # 1 * h * w

    x1, y1, w1, h1 = box_1
    x2, y2, w2, h2 = box_2
    x1, x2, y1, y2, w1, w2, h1, h2 = int(x1), int(x2), int(y1), int(y2), int(w1), int(w2), int(h1), int(h2)
    print(f'mask_1.shape = {mask_1.shape}, mask_2.shape = {mask_2.shape}')
    print(f'x1 = {x1}, y1 = {y1}, w1 = {w1}, h1 = {h1}')
    print(f'x2 = {x2}, y2 = {y2}, w2 = {w2}, h2 = {h2}')
    valid_mask = mask_1[y1:y1+h1,x1:x1+w1].unsqueeze(0) # [1, h1, w1]
    valid_mask = rearrange(valid_mask, 'c h w -> 1 c h w')
    # print(f'valid_mask.shape = {valid_mask.shape}')
    resized_valid_mask = F.interpolate(
        valid_mask,
        size=(h2, w2),
        mode='bilinear',
        align_corners=False
    ) # [1, 1, h2, w2]
    # resized_valid_mask = rearrange(repeat(rearrange(resized_valid_mask, 'h w -> 1 h w'), '1 h w -> c h w', c=3), 'c h w -> 1 c h w')
    # resized_valid_mask = resized_valid_mask.unsqueeze(0)
    resized_valid_mask[resized_valid_mask > 0.5] = 1.
    resized_valid_mask[resized_valid_mask <= 0.5] = 0.
    # print(f'resized_valid_mask.shape = {resized_valid_mask.shape}') # 1 * h * w
    # print(f'part: mask_2[:, y2:y2+h2, x2:x2+w2].shape = {mask_2[:, y2:y2+h2, x2:x2+w2].shape}') # 1 * w * h
    mask_2[:,y2:y2+h2,x2:x2+w2] = resized_valid_mask[0,:,:,:]
    assert mask_2.squeeze().shape == mask_1.squeeze().shape, f'mask_1.shape = {mask_1.shape}, mask_2.shape = {mask_2.shape}'
    return repeat(mask_2, '1 h w -> c h w', c = 3).unsqueeze(0) if type == 'remove' else mask_2
    # 1 * 3 * h * w

# TODO: implement 'refact_target'

def mask_inside_box(target_box, mask, erode_kernel, erode_iteration):
    # mask: [1, 3, w, h]
    mask_out = torch.zeros_like(mask, dtype=torch.uint8)
    x, y, w, h = target_box
    for i in range(x, x+w):
        for j in range(y, y+h):
            mask_out[i,j] = 255
    mask_out = cv2.erode(np.uint8(mask_out), erode_kernel, iterations=erode_iteration)
    return mask_out

def fix_box(gpt_box, img_shape, ori_box=None):
    assert len(gpt_box) == 4 and len(img_shape) == 3
    x, y, w, h = gpt_box
    fixed_box = (0,0,0,0)
    if w < 0: 
        w = -w
        print('?')
    if h < 0: 
        h = -h
        print('?')
    if h == 0 or w == 0:
        return fixed_box
    
    H_, W_, _ = img_shape # [h, w, 3]
    flag = 0
    while x + w >= W_ or y + h >= H_:
        flag += 1
        if flag > 4:
            return fixed_box
        if x < W_ and y < H_:
            print(f'1-Fixing: with (W,H) = {(W_, H_)}')
            w //= 2
            h //= 2
        else:
            print(f'2-Fixing: with (x,y) = {(x, y)}')
            x //= 2
            y //= 2
    if ori_box is not None:
        h2, w2 = ori_box[2], ori_box[3]
        if randint(0,1) == 1:
            print('-'*6 + ' Froze: W ' + '-'*6)
            h = h2/w2 * w
        else:
            print('-'*6 + ' Froze: H ' + '-'*6)
            w = w2/h2 * h
        return (x, y, w, h)
    else:
        return (x, y, w, h)
