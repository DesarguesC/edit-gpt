import torch, os, request
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from functools import partial
from datasets import load_dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from zipfile import ZipFile
from PIL import Image
from basicsr.utils import tensor2img, img2tensor


"""
    FID results tend to be fragile as they depend on a lot of factors:
        The specific Inception model used during computation.
        The implementation accuracy of the computation.
        The image format (not the same if we start from PNGs vs JPGs).
    Keeping that in mind, FID is often most useful when comparing similar runs, 
    but it is hard to reproduce paper results unless the authors carefully disclose the FID measurement code.
    These points apply to other related metrics too, such as KID and IS.
"""

def Init_FID(real_image_list):
    assert isinstance(real_image_list, list)
    # init fid class with list real_images
    # real_image: list(np.array), with the same shape (512,512) or (256,256)
    real_images = torch.cat([tensor2img(img) for img in real_image_list])
    fid = FID(normalize=True)
    fid.update(real_images, real=True)
    return fid


def Cal_FIDScore(fid_inited, fake_image):
    fid_inited.update(fake_images, real=False)
    return fid.compute()


