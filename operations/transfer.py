import numpy as np
import torch, cv2, os
from paint.bgutils import target_removing
from paint.crutils import ab64
# calculate IoU between SAM & SEEM
from PIL import Image
from einops import repeat, rearrange
from paint.bgutils import refactor_mask, match_sam_box
from paint.utils import (recover_size, resize_and_pad, load_img_to_array, save_array_to_img, dilate_mask)
from operations.utils import inpaint_img_with_lama
from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything

def Transfer_Me_ip2p(opt, dilate_kernel_size):
    