import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import argparse
import whisper
import numpy as np

from gradio import processing_utils
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from demo.seem.tasks import *

conf_files = "configs/seem/focall_unicl_lang_demo.yaml"

opt = load_opt_from_config_files([conf_files])
opt = init_distributed(opt)


cur_model = 'None'
pretrained_pth = '~/autodl-tmp'

if 'focalt' in conf_files:
    pretrained_pth = os.path.join(pretrained_pth, "seem_focalt_v0.pt")
    cur_model = 'Focal-T'
elif 'focall' in conf_files:
    pretrained_pth = os.path.join(pretrained_pth, "seem_focall_v0.pt")
    cur_model = 'Focal-L'
    # this

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

audio = whisper.load_model("base")

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
        else:
            return interactive_infer_image(model, audio, image, task, *args, **kwargs)


from PIL import Image
input_image = Image.open('../2.jpg')

image = interactive_infer_image(model, audio, input_image, ['Task'])
print(image)



