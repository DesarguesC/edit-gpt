import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from seem.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

from seem.utils.arguments import load_opt_from_config_files
from seem.modeling.BaseModel import BaseModel
from seem.modeling import build_model
from seem.utils.constants import COCO_PANOPTIC_CLASSES
from seem.demo.seem.tasks import *



def query_middleware(opt, image: Image, reftxt: str):
    # target-noun => mask & box

    # image: removed pil image
    # reftxt: query target text
    cfg = load_opt_from_config_files([opt.seem_cfg])
    cfg['device'] = opt.device
    seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(opt.seem_ckpt).eval().cuda()
    with torch.no_grad():
        seem_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"],
                                                                                 is_eval=True)
    # get text-image mask
    image_ori = transform(image)
    width, height = image_ori.size
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    data = {"image": images, "height": height, "width": width}
    visual = Visualizer(image_ori, metadata=metadata)

    seem_model.model.task_switch['spatial'] = False
    seem_model.model.task_switch['visual'] = False
    seem_model.model.task_switch['grounding'] = False
    seem_model.model.task_switch['audio'] = False
    seem_model.model.task_switch['grounding'] = True

    data['text'] = [reftxt]
    batch_inputs = [data]



    results, image_size, extra = seem_model.model.evaluate_demo(batch_inputs)
    print(f'extra.keys() = {extra.keys()}')
    print(f'results.keys() = {results.keys()}')

    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = seem_model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id,:,:]
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    # mask queried from text

    pred_box_pos = ?

    demo = visual.draw_panoptic_seg(mask_all.cpu(), category)  # rgb Image
    res = demo.get_image()

    return Image.fromarray(res), pred_masks_pos, pred_box_pos


def middleware(opt, image: Image, reftxt: str, visual_mode=True):
    # image: target removed PIL image

    cfg = load_opt_from_config_files([opt.seem_cfg])
    cfg['device'] = opt.device
    seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(opt.seem_ckpt).eval().cuda()
    with torch.no_grad():
        seem_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    image_ori = transform(image)
    width, height = image_ori.size
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    if visual_mode:
        visual = Visualizer(image_ori, metadata=metadata)
    data = {"image": images, "height": height, "width": width}

    # inistalize task
    seem_model.model.task_switch['spatial'] = False
    seem_model.model.task_switch['visual'] = False
    seem_model.model.task_switch['grounding'] = False
    seem_model.model.task_switch['audio'] = False
    seem_model.model.task_switch['grounding'] = True

    data['text'] = [reftxt]
    batch_inputs = [data]

    # if 'Panoptic' in tasks:
    seem_model.model.metadata = metadata
    results, mask_box_dict = seem_model.model.evaluate_all(batch_inputs)
    mask_all, category, masks_list = results[-1]['panoptic_seg']

    assert len(category) == len(masks_list), f'len(category) = {len(category)}, len(masks_list) = {len(masks_list)}'
    object_mask_list = [ {
        'name': metadata.stuff_classes[category[i]['category_id']],
        'mask': masks_list[i]
    }  for i in range(len(category)) ]

    demo = visual.draw_panoptic_seg(mask_all.cpu(), category) # rgb Image
    res = demo.get_image()

    return Image.fromarray(res), object_mask_list

