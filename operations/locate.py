from operations.remove import Remove_Me, Remove_Me_lama
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

def re_mask(mask, dtype=torch.float32):
    mask[mask >= 0.5] = 1.
    mask[mask < 0.5] = 0.
    return torch.tensor(mask, dtype=dtype)

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
    question = f'Size: ({opt.W},{opt.H})\n' + question
    print(f'question: {question}')
    box_ans = get_response(edit_agent, question)
    print(f'box_ans = {box_ans}')

    # deal with the answer, procedure is the same as in replace.py
    box_ans = box_ans.split('[')[-1].split(']')[0]
    punctuation = re.compile('[^\w\s]+')
    box_ans = [x.strip() for x in re.split(punctuation, box_ans) if x != ' ' and x != '']
    x, y, w, h = int(box_ans[1]), int(box_ans[2]), int(box_ans[3]), int(box_ans[4])
    box_0 = (x, y, w, h)
    destination_mask = refactor_mask(target_box, target_mask, box_0)
    target_mask, destination_mask = re_mask(target_mask), re_mask(destination_mask)
    
    print(f'target_mask.shape = {target_mask.shape}, destination_mask.shape = {destination_mask.shape}')
    
    cv2.imwrite(f'./outputs/destination-mask-{opt.out_name}', cv2.cvtColor(np.uint8(255. * rearrange(repeat(destination_mask.squeeze(0),\
                                                        '1 ... -> c ...', c=3), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'./outputs/target-mask-{opt.out_name}', cv2.cvtColor(np.uint8(255. * rearrange(repeat(target_mask,\
                                                        '1 ... -> c ...', c=3), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    print(f'destination-mask saved: \'./outputs/destination-mask-{opt.out_name}\'')
    print(f'destination-mask saved: \'./outputs/target-mask-{opt.out_name}\'')
    
    print(f'np.array(img_pil).shape = {np.array(img_pil).shape}')
    xt, yt, wt, ht = target_box
    yt_ = (yt+ht) if yt+ht < opt.H else int(abs(yt-ht))
    xt_ = (xt+wt) if xt+wt < opt.W else int(abs(xt-wt))
    Ref_Image = (np.array(img_pil)*np.array(target_mask))[min(yt,yt_):max(yt,yt_), min(xt,xt_):max(xt,xt_), :]
    print(f'Ref_Image.shape = {Ref_Image.shape}')
    cv2.imwrite('./static/Ref-location.jpg', cv2.cvtColor(np.uint8(Ref_Image), cv2.COLOR_BGR2RGB))
    # SAVE_TEST
    print(f'Ref_Image.shape = {Ref_Image.shape}, target_mask.shape = {target_mask.shape}')
    Ref_Image = Ref_Image * repeat(rearrange(target_mask.cpu().detach().numpy()\
                                                 [:,min(yt,yt_):max(yt,yt_), min(xt,xt_):max(xt,xt_)], 'c h w -> h w c'), '... 1 -> ... c', c=3)
    output_path, x_sample_ddim = paint_by_example(opt, destination_mask, Image.fromarray(np.uint8(Ref_Image)), rm_img)
    # use rm_img or original img_pil ?
    cv2.imwrite(output_path, tensor2img(x_sample_ddim))

    print('exit from create_location')
    exit(0)


