import pdb
import cv2
import os
from collections import OrderedDict

import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, url_for, render_template, request, redirect, send_from_directory
from PIL import Image
import base64
import io
import random

from options.test_options import TestOptions
import models # from crfill
import torch

max_size = 256
max_num_examples = 200


def ab64(x):
    if (x-x//64*64)<32: return x//64*64
    else: return (x//64+1)*64

def ab8(x):
    if (x-x//8*8)<4: return x//8*8
    else: return (x//8+1)*8



def get_crfill_model(opt):
    print('getting model')
    crfill_model = models.create_model(opt)
    print('getting model done')
    crfill_model.eval()
    return crfill_model



def process_image_via_crfill(img, mask, opt, crfill_model=None):
    if crfill_model == None: crfill_model = get_crfill_model(opt)
    assert crfill_model != None , 'crfill_model remained None!'
    
    """
    
    img =img.convert("RGB")
    img_raw = np.array(img)
    # img: np.array(np.uint8)
    print(f'img.size = {img.size}') # size or shape ?
    w_raw, h_raw = img.size
    h_t, w_t = ab8(h_raw), ab8(w_raw)

    img = img.resize((w_t, h_t))
    img = np.array(img).transpose((2,0,1))
    print(f'resized img.shape = {img.shape}')

    mask_raw = np.array(mask)[...,None]>0
    mask = mask.resize((w_t, h_t))
    print(f'resized mask.shape = {mask.shape}')

    mask = np.array(mask)
    mask = (torch.Tensor(mask)>0).float()
    img = (torch.Tensor(img)).float()
    img = 2. * img/255. - 1.
    img = img[None]
    mask = mask[None,None]
    
    """

    mask = torch.tensor(mask, dtype=torch.float32, requires_grad=False)
    img = 2. * torch.tensor(img / 255., dtype=torch.float32, requires_grad=False) - 1.
    
    print('in')
    with torch.no_grad():
        generated,_ = crfill_model({'image':img, 'mask':mask}, mode='inference')
    print('out')
    
    generated = torch.clamp(generated, -1, 1)
    generated = (generated + 1.) / 2. * 255.
    generated = generated.cpu().numpy().astype(np.uint8)
    generated = generated[0].transpose((1,2,0))
    result = generated * mask + img * (1. - mask)
    result = result.astype(np.uint8)

    result = Image.fromarray(result).resize((w_raw, h_raw))
    result = np.array(result)
    result = Image.fromarray(result.astype(np.uint8))
    return result
    # result.save(f"static/removed/{opt.name}")
    



