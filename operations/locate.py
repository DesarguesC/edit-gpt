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

def create_location(opt, target, edit_agent=None):
    assert edit_agent != None, 'no edit agent'
    # move the target to the destination, editing via GPT (tell the bounding box)
    img_pil = Image.open(opt.in_dir).convert('RGB')
    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))
    # resize and prepare the original image

    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.to(device=opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    # prepare SAM, matched via SEEM
    mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)), \
                           key=(lambda x: x['area']), reverse=True)
    sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list]


    # remove the target, get the mask (for bbox searching via SAM)
    rm_img, target_mask, _ = RM(opt, target, remove_mask=True)
    rm_img = Image.fromarray(cv2.cvtColor(rm_img, cv2.COLOR_RGB2BGR)).convert('RGB')
    res, panoptic_dict = middleware(opt, rm_img)  # key: name, mask

    # destination: {[name, (x,y), (w,h)], ...} + edit-txt (tell GPT to find the target noun) + seg-box (as a hint) ==>  new box
    target_box = match_sam_box(target_mask, sam_seg_list)  # target box
    bbox_list = [match_sam_box(x['mask'], sam_seg_list) for x in panoptic_dict]
    print(target_box)
    print(f'bbox_list: {bbox_list}')

    box_name_list = [{
        'name': panoptic_dict[i]['name'],
        'bbox': bbox_list[i]
    } for i in range(len(bbox_list))]
    box_name_list.append({
        'name': target,
        'bbox': target_box
    }) # as a hint

    question = Label().get_str_location(box_name_list, opt.edit_txt) # => [name, (x,y), (w,h)]
    box_ans = get_response(edit_agent, question)
    print(f'box_ans = {box_ans}')

    # deal with the answer, procedure is the same as in replace.py
    box_ans = box_ans.split('[')[-1].split(']')[0]
    punctuation = re.compile('[^\w\s]+')
    box_ans = [x.strip() for x in re.split(punctuation, box_ans) if x != ' ' and x != '']
    x, y, w, h = int(box_ans[1]), int(box_ans[2]), int(box_ans[3]), int(box_ans[4])
    box_0 = (x, y, w, h)
    destination_mask = refactor_mask(target_box, target_mask, box_0)

    Ref_Image = cv2.cvtColor(np.array(img_pil)[y:y+h, x:x+w, :], cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/Ref-location.jpg', cv2.cvtColor(Ref_Image, cv2.COLOR_BGR2RGB))
    # SAVE_TEST
    print(f'Ref_Image.shape = {Ref_Image.shape}, target_mask.shape = {target_mask.shape}')
    Ref_Image = Ref_Image * target_mask
    output_path, x_sample_ddim = paint_by_example(opt, destination_mask, Image.fromarray(Ref_Image), img_pil, use_adapter=opt.use_adapter)
    cv2.imwrite(output_path, tensor2img(x_sample_ddim))

    print('exit from create_location')
    exit(0)


