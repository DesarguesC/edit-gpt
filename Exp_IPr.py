import os, time, json, logging

from basicsr.utils import tensor2img, img2tensor
from random import randint
from PIL import Image, ImageOps
import numpy as np

from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import calculate_clip_score, PSNR_compute, SSIM_compute, write_instruction, write_valuation_results, cal_metrics_write
from preload_utils import *
from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial
from pytorch_lightning import seed_everything



if __name__ == '__main__':
    start_time = time.time()
    opt = get_arguments()

    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')