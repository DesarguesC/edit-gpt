import os, sys, torch
import numpy as np
import argparse
import whisper
from PIL import Image
import cv2

from modeling.BaseModel import BaseModel
from modeling import build_model
# from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from demo.seem.tasks import *


def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", help='path to config file', )
    # set as default
    parser.add_argument('--in_dir', default='../autodl-tmp/assets/inputs/1.jpg', help='path to input image file')
    parser.add_argument('--out_dir', default='../autodl-tmp/assets/outputs', help='path to output image file')
    parser.add_argument('--name', default='1.jpg', help='output image name')
    parser.add_argument('--reftxt', default='everything', help='prompts')

    cfg = parser.parse_args()
    return cfg


cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files]) # opt -> dict
# opt = init_distributed(opt)

if 'device' not in opt.keys():
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


pretrained_pth = '../autodl-tmp/seem_focall_v0.pt'
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
# seem base model

with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
# label

# audio_model = whisper.load_model('base')
# audio model => useless in my project?


img_pil, _ = interactive_infer_image(model, None, Image.open(cfg.in_dir), ['Text'], None, cfg.reftxt, None, None)


cfg.name = cfg.in_dir.spilit('/')[-1] if cfg.name == None else cfg.name
img_pil.save(os.path.join(cfg.out_dir, cfg.name))

# SEEM failed to understand the meaning of the number