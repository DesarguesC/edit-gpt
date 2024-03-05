from operations.remove import Remove_Me, Remove_Me_lama
from seem.masks import middleware
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
from paint.example import paint_by_example
from paint.bgutils import match_sam_box, refactor_mask, fix_box
from PIL import Image
import numpy as np
# from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog as mt
import torch, cv2, os
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
TURN = lambda x: repeat(rearrange(np.array(x.cpu().detach().numpy()), 'c h w -> h w c'), '... 1 -> ... c', c=3)

def re_mask(mask, dtype=torch.float32):
    mask[mask >= 0.5] = 1.
    mask[mask < 0.5] = 0.
    return torch.tensor(mask, dtype=dtype)

def get_area(ref_img, box):
    assert len(box) == 4
    # if box cooresponds to an area out of the ref image, then move back the box
    x, y, w, h = box
    x1, y1 = x + w, y + h
    return ref_img[y:y1, x:x1, :]
    
def fine_box(box, img_shape):
    # TODO: fine a box into ori-img area
    pass
    

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
    rm_img, target_mask, _ = Remove_Me_lama(opt, target, dilate_kernel_size=opt.dilate_kernel_size) if opt.use_lama \
                        else Remove_Me(opt, target, remove_mask=True, replace_box=opt.replace_box)
    rm_img = Image.fromarray(rm_img)
    # rm_img.save(f'./static-inpaint/rm-img.jpg')
    # print(f'removed image saved at \'./static-inpaint/rm-img.jpg\'')

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

    question = Label().get_str_location(box_name_list, opt.edit_txt, (opt.W,opt.H)) # => [name, (x,y), (w,h)]
    # question = f'Size: ({opt.W},{opt.H})\n' + question
    print(f'question: \n{question}')

    box_0 = (0,0,0,0)
    try_time = 0

    while box_0 == (0,0,0,0):
        if try_time > 0:
            print(f'Trying to fix... - Iter: {try_time}')
        box_ans = get_response(edit_agent, question)
        print(f'box_ans = {box_ans}')
        # deal with the answer, procedure is the same as in replace.py
        box_ans = box_ans.split('[')[-1].split(']')[0]
        punctuation = re.compile('[^\w\s]+')
        box_ans = [x.strip() for x in re.split(punctuation, box_ans) if x != ' ' and x != '']
        x, y, w, h = int(box_ans[1]), int(box_ans[2]), int(box_ans[3]), int(box_ans[4])
        box_0 = (x, y, int(w * opt.expand_scale), int(h * opt.expand_scale))
        box_0 = fix_box(box_0, (opt.W,opt.H,3))
        
        try_time += 1

    print(f'BEFORE: box_0={box_0}')
    
    destination_mask = refactor_mask(target_box, target_mask, box_0)
    target_mask, destination_mask = re_mask(target_mask), re_mask(destination_mask)
    if torch.max(target_mask) <= 1.:
        target_mask[target_mask > 0.5] = 0.95 if opt.mask_ablation else 1.
        target_mask[target_mask <= 0.5] = 0.05 if opt.mask_ablation else 0.
    
    os.mkdir(opt.mask_dir)
    print(f'target_mask.shape = {target_mask.shape}, destination_mask.shape = {destination_mask.shape}')
    # target_mask: [1, h, w], destination_mask: [1, 3, h, w]
    cv2.imwrite(f'./{opt.mask_dir}/destination.jpg', cv2.cvtColor(np.uint8((255. if torch.max(destination_mask) <= 1. else 1.) \
                            * rearrange(destination_mask.squeeze(0), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'./{opt.mask_dir}/target.jpg', cv2.cvtColor(np.uint8((255. if torch.max(target_mask) <= 1. else 1.) \
                            * rearrange(repeat(target_mask, '1 ... -> c ...', c=3), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    print(f'destination-mask saved: \'./{opt.mask_dir}/destination.jpg\'') # 目的位置的mask (编辑后)
    print(f'destination-mask saved: \'./{opt.mask_dir}/target.jpg\'') # 原始图片的mask (编辑前)

    # TODO: Validate destination_mask content
    # d1, d2, d3 = destination_mask[0].chunk(3)
    # print(d1==d2)
    # print(d2==d3)
    
    img_np = np.array(img_pil)
    Ref_Image = get_area(img_np * TURN(target_mask), target_box)

    print(f'img_np.shape = {img_np.shape}, Ref_Image.shape = {Ref_Image.shape}')
    # cv2.imwrite('./static/Ref-location.jpg', cv2.cvtColor(np.uint8(Ref_Image), cv2.COLOR_BGR2RGB))
    # SAVE_TEST
    print(f'Ref_Image.shape = {Ref_Image.shape}, target_mask.shape = {target_mask.shape}')
    op_output, x_sample_ddim = paint_by_example(opt, destination_mask, Image.fromarray(np.uint8(Ref_Image)), rm_img)
    print(f'x_sample_ddim.shape = {x_sample_ddim.shape}, TURN(target_mask).shape = {TURN(target_mask).shape}, img_np.shape = {img_np.shape}')

    x_sample_ddim = tensor2img(x_sample_ddim)
    cv2.imwrite(op_output, x_sample_ddim)
    print(f'locate result image saved at \'{op_output}\'')
    # TODO: If I need to Paint-by-Example ?

    print('exit from create_location')
    exit(0)


