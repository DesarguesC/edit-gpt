from operations.remove import Remove_Me as RM
from seem.masks import middleware
from paint.crutils import ab8, ab64
from paint.bgutils import refactor_mask, match_sam_box
from paint.example import paint_by_example
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


def find_boxes_for_masks(masks: torch.tensor, nouns: list[str], sam_list: list[tuple]):
    # masks: segmentation gained from SEEM
    # nouns: segmentation nouns gained from SEEM
    # assert len(nouns) == len(masks), f'len(nouns) = {len(nouns)}, len(masks) = {len(masks)}'
    
    print(f'nouns = {nouns}')
    print(f'masks.shape = {masks.shape}')
    # sam_list: [mask, box] list gained from SAM
    seem_dict = {}
    # {'name': (mask, box)}
    assert len(masks) <= len(sam_list)
    for i in range(len(nouns)):
        name = nouns[i]
        mask = masks[i]
        box_idx = np.argmax([np.sum(mask.cpu().detach().numpy() * sam_out[1], keep_dim=False) for sam_out in sam_list])
        seem_dict[name] = [mask, sam_list[box_idx][0]]
        del sam_list[box_idx]
    return seem_dict




def preprocess_image2mask(opt, old_noun, new_noun, img: Image, diffusion_img: Image):
    
    """
        img => original image
        diffusion_img => example image
        
        TODO: query
            
            1. img (original image): query <old_noun> => (box_1, mask_1)
            2. remove <old_noun> from original image => rm_img
            3. rm_img (removed image): Panoptic => object-list
            4. diffusion_img: query <new_noun> => (box_2, mask_2)
            5. GPT3.5: rescale (box_2, mask_2) according to object-list
            6. Paint-by-Example: paint [difusion_img] to [rm_img], constrained by [mask_2]
    """
    
    # deal with the image removed target noun
    res_all, objects_masks_list = middleware(opt=opt, image=img, diffusion_image=diffusion_image)
    res_all.save('./tmp/panoptic-seg.png')
    return res_all, objects_masks_list





def generate_example(opt, new_noun) -> Image:
    sd_model, sd_sampler = get_sd_models(opt)
    with torch.inference_mode(),  sd_model.ema_scope(), autocast('cuda'):
        seed_everything(opt.seed)
        diffusion_image = diffusion_inference(opt, new_noun, sd_model, sd_sampler)
        diffusion_image = tensor2img(diffusion_image)
        cv2.imwrite('./static/ref.jpg', diffusion_image)
        print(f'example saved at \'./static/ref.jpg\'')
        diffusion_image = cv2.cvtColor(diffusion_image, cv2.COLOR_BGR2RGB)
    del sd_model, sd_sampler
    return Image.fromarray(diffusion_image) # pil


def replace_target(opt, old_noun, new_noun, label_done=None, edit_agent=None):
    # assert mask_generator != None, 'mask_generator not initialized'
    assert edit_agent != None, 'no edit agent!'
    img_pil = Image.open(opt.in_dir).convert('RGB')
    
    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))
     
    # old_noun
    # res, mask_1, _ = query_middleware(opt, img_pil, old_noun) # not sure if it can get box for single target
    
    rm_img, mask_1, _ = RM(opt, old_noun, remove_mask=True)
    rm_img = Image.fromarray(cv2.cvtColor(rm_img, cv2.COLOR_RGB2BGR))
    
    res, panoptic_dict = middleware(opt, rm_img) # key: name, mask
    diffusion_pil = generate_example(opt, new_noun)
    _, mask_2, _ = query_middleware(opt, diffusion_pil, new_noun)
    
    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.to(device=opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    mask_box_list = mask_generator.generate(np.array(img_pil))
    mask_box_list = sorted(mask_box_list, key=(lambda x: x['area']), reverse=True)
    # print(f'mask_box_list[0].keys() = {mask_box_list[0].keys()}')
    sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list]
    box_1 = match_sam_box(mask_1, sam_seg_list) # old noun
    bbox_list = [match_sam_box(x['mask'], sam_seg_list) for x in panoptic_dict]
    print(box_1)
    print(bbox_list)
       
    box_name_list = [{
        'name': panoptic_dict[i]['name'],
        'bbox': bbox_list[i]
    } for i in range(len(bbox_list))]
    box_name_list.append({
        'name': old_noun,
        'bbox': box_1
    })
    
    diffusion_mask_box_list = sorted(mask_generator.generate(np.array(diffusion_pil)), key=(lambda x: x['area']), reverse=True)
    box_2 = match_sam_box(mask_2, [(u['bbox'], u['segmentation'], u['area']) for u in diffusion_mask_box_list])
    
    
    question = Label().get_str_rescale(old_noun, new_noun, box_name_list)
    print(f'question: {question}')
    ans = get_response(edit_agent, question)
    print(f'ans: {ans}')
    
    box_0 = ans.split('[')[-1].split(']')[0]
    punctuation = re.compile('[^\w\s]+')
    box_0 = re.split(punctuation, box_0)
    box_0 = [x for x in box_0 if x!= ' ' and x!='']
    # print(f'box_0 = {box_0}')
    # print(f'len(box_0) = {len(box_0)}') # length = 5
    
    new_noun, x, y, w, h = box_0[0], box_0[1], box_0[2], box_0[3], box_0[4]
    new_noun, x, y, w, h = new_noun.strip(), x.strip(), y.strip(), w.strip(), h.strip()
    print(f'new_noun, x, y, w, h = {new_noun}, {x}, {y}, {w}, {h}')
    box_0 = (int(x), int(y), int(w), int(h))
    target_mask = refactor_mask(box_2, mask_2, box_0)
    print(f'target_mask.shape = {target_mask.shape}')
    
    output_path, results = paint_by_example(opt, mask=target_mask, ref_img=diffusion_pil, base_img=rm_img)
    cv2.imwrite(output_path, tensor2img(results))
    print('exit before rescaling')
    exit(0)
    
    
    # seg_res, objects_masks_list = preprocess_image2mask(opt, img_pil, diffusion_pil)

    print('exit from replace_target')
    eixt(0)

    removed_np, target_mask, target_box, *_ = RM(opt, old_noun, remove_mask=True)
    removed_pil = Image.fromarray(removed_np)
    seg_res, objects_masks_list = preprocess_image2mask(opt, removed_pil)
    
    sam_masks = mask_generator.generate(np.array(seg_res))
    sam_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
    print(f'a box: {sam_list[0][0]}')
    
    for i in range(len(objects_masks_list)):
        mask_ = objects_masks_list[i]['mask']
        objects_masks_list[i]['bbox'], sam_list = match_sam_box(mask_, sam_list)
    
    """
        objects_masks_list = [
            {'name': ..., 'mask': ..., 'bbox': ...},
            ...
        ]
    
    """
    # print('exit')
    # exit(0)
    
    # TODO: <0> create [name, (x,y,w,h)] list to ask GPT-3.5 and arrange a place for [new_noun, (x,y,w,h)]
    series = Label().get_str_part(objects_masks_list)
    print(series)
    # from revChatGPT.V3 import Chatbot # add new_noun via edit_agent
    """
        e.g. new_noun = 'cat'
    """
    edit_prompt = "Objects: " + series + "; New: " + new_noun
    new = get_response(edit_agent, edit_prompt)
    print(new)
    w_h = re.split('[{}(),]', new.strip('[').strip(']'))
    w, h = w_h # gain new target noun sizes (w,h)
    print(w, h)

    # TODO: <1> create LIST
    # TODO: <2> apply an agent to generate [new_noun, (x,y,w,h)] ~ [mask]
    # TODO: <3> add some prompts to generate an image (restore required) for new_noun (via diffusion) and extract [mask, box] via SEEM
    # TODO: <4> rescale the mask and the box
    # TODO: <5> Paint-by-Example using the [mask, image] above




