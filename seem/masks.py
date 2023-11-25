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

def middleware(opt, image, reftxt, tasks=['Text']):
    # mask_cover: [0,0,0] -> cover mask area via black
    cfg = load_opt_from_config_files([opt.seem_cfg])
    cfg['device'] = opt.device
    seem_model = BaseModel(cfg, build_model(cfg)).from_pretrained(opt.seem_ckpt).eval().cuda() 
    with torch.no_grad():
        seem_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    image_ori = transform(image)
    width = image_ori.size[0]
    height = image_ori.size[1]
    print(1)
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    data = {"image": images, "height": height, "width": width}
    # if len(tasks) == 0:
    #     tasks = ["Panoptic"]
    
    # inistalize task
    seem_model.model.task_switch['spatial'] = False
    seem_model.model.task_switch['visual'] = False
    seem_model.model.task_switch['grounding'] = False
    seem_model.model.task_switch['audio'] = False
    example = None
    
    seem_model.model.task_switch['grounding'] = True
    data['text'] = [reftxt]
    batch_inputs = [data]
    results, image_size, extra = seem_model.model.evaluate_demo(batch_inputs)
    
    pred_masks = results['pred_masks'][0]
    print(f'results.keys() = {results.keys()}')
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = seem_model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id,:,:]
    pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]
    
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    # mask
    texts = [all_classes[pred_class[0]]]

    for idx, mask in enumerate(pred_masks_pos):
        # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
        out_txt = texts[idx] # if 'Text' not in tasks else reftxt
        demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0]%133], text=out_txt)
    res = demo.get_image()
    torch.cuda.empty_cache()
    return res, pred_masks_pos

