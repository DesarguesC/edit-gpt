import os, time, json, logging

import os, time, json, logging
from random import randint, choice
from PIL import Image, ImageOps
from socket import *
import numpy as np
from task_planning import Replace_Method, Move_Method, Transfer_Method
from operations.vqa_utils import A_IsReplacedWith_B, preload_vqa_model
from prompt.guide import get_response, get_bot, system_prompt_gen_move_instructions, system_prompt_edit_sort
from task_planning import Add_Method, Remove_Method, Transfer_Method
from prompt.arguments import get_arguments
from prompt.util import write_instruction, write_valuation_results, cal_metrics_write
from preload_utils import *
from operations.vqa_utils import preload_vqa_model, Val_add_amount, IsRemoved
from pytorch_lightning import seed_everything

from prompt.arguments import get_arguments
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