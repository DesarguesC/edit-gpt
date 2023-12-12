from operations.remove import Remove_Me as RM
from seem.masks import middleware
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
from paint.example import paint_by_example
from paint.bgutils import match_sam_box, refactor_mask
from PIL import Image
import numpy as np
# from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog as mt
import torch, cv2
from torch import autocast
from torch.nn import functional as F
from prompt.item import Label
from prompt.guide import get_response
from jieba import re
from seem.masks import middleware, query_middleware
from ldm.inference_base import diffusion_inference, get_sd_models
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
from einops import repeat, rearrange

def create_location(opt, target, destination, edit_agent=None):
    assert edit_agent != None, 'no edit agent'
    # move the target to the destination
    img_pil = Image.open(opt.in_dir).convert('RGB')

    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))

    # remove the target, get the mask (for bbox searching via SAM)
    rm_img, target_mask, _ = RM(opt, target, remove_mask=True)
    rm_img = Image.fromarray(cv2.cvtColor(rm_img, cv2.COLOR_RGB2BGR))

    res, panoptic_dict = middleware(opt, rm_img)  # key: name, mask
    # _, target_mask, _ = query_middleware(opt, img_pil, target)
    _, destination_mask, _ = query_middleware(opt, img_pil, destination)

    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.to(device=opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    mask_box_list = mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
    mask_box_list = sorted(mask_box_list, key=(lambda x: x['area']), reverse=True)

    sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list]
    box_target = match_sam_box(target_mask, sam_seg_list)  # target box
    box_destination = match_sam_box(destination_mask, sam_seg_list)
    bbox_list = [match_sam_box(x['mask'], sam_seg_list) for x in panoptic_dict]
    print(box_target)
    print(bbox_list)

    box_name_list = [{
        'name': panoptic_dict[i]['name'],
        'bbox': bbox_list[i]
    } for i in range(len(bbox_list))]
    box_name_list.append({
        'name': target,
        'bbox': box_target
    })

    # generate a new box where to place the target
    question = Label().get_str_location(box_name_list, destination)
    # remain to implement in Label method
    box_ans = get_response(edit_agent, question)
    # '[x, y, w, h]'
    box_ans = box_ans.strip('[').strip(']').split(',')
    assert len(box_ans) == 4, f'box_ans = {box_ans}'
    box_ans = (box_ans[0], box_ans[1], box_ans[2], box_ans[3])
    mask_destination = refactor_mask(box_target, target_mask, box_ans)

    np_img = np.array(img_pil)
    print(f'np_img.shape = {np_img.shape}')
    x0, y0, w0, h0 = box_target
    Ref_Image = cv2.cvtColor(np_img[y0:y0+h0, x0:x0+w0, :], cv2.COLOR_BGR2RGB)
    output_path, x_sample_ddim = paint_by_example(opt, mask_destination, Image.fromarray(Ref_Image), img_pil)
    cv2.imwrite(output_path, tensor2img(x_sample_ddim))

    print('exit from create_location')
    exit(0)

