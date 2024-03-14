from operations import Remove_Me, Remove_Me_lama
from seem.masks import middleware
from paint.crutils import ab8, ab64
from paint.bgutils import refactor_mask, match_sam_box, fix_box
from paint.example import paint_by_example
from PIL import Image
import numpy as np
# from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog as mt
from torch import autocast
from torch.nn import functional as F
from prompt.item import Label
from prompt.guide import get_response, Use_Agent
from jieba import re
from seem.masks import middleware, query_middleware
from ldm.inference_base import *
from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
from einops import repeat, rearrange
import torch, cv2, os
from paint.example import generate_example
from operations.utils import get_reshaped_img
from operations.add import Add_Object


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
    res_all, objects_masks_list = middleware(opt=opt, image=img, diffusion_image=diffusion_img)
    res_all.save('./tmp/panoptic-seg.png')
    return res_all, objects_masks_list

def replace_target(
        opt, 
        old_noun: str, 
        new_noun: str, 
        input_pil: Image = None, 
        edit_agent = None, 
        expand_agent = None,
        preloaded_model = None
    ):
    """
    -> preloaded_model
        Keys in need:
            preloaded_lama_remover
            preloaded_seem_detector
            preloaded_sam_generator
            preloaded_example_generator
            preloaded_example_painter
    """

    # assert mask_generator != None, 'mask_generator not initialized'
    assert edit_agent != None, 'no edit agent!'
    opt, img_pil = get_reshaped_img(opt, input_pil)
    
    rm_img, mask_1, _ = Remove_Me_lama(
                            opt, old_noun, dilate_kernel_size = opt.dilate_kernel_size, 
                            intput_pil = input_pil, preloaded_model = preloaded_model
                        ) if opt.use_lama \
                        else Remove_Me(opt, old_noun, remove_mask=True, replace_box=opt.replace_box)
    
    rm_img = Image.fromarray(rm_img)
    _, panoptic_dict = middleware(
                            opt, rm_img, 
                            preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                        ) # key: name, mask
    print(f'total names: ', panoptic_dict[0]['name'])
    if old_noun not in panoptic_dict[0]['name']:
        print(f'find no \'{old_noun}\' in panoptic_dict of input, Start Add process...')
        return Add_Object(
                    opt, 
                    name = new_noun, 
                    num = 1,
                    place = '<NULL>',
                    input_pil=img_pil,
                    edit_agent = Use_Agent(opt, TODO='adjust bbox for me'),
                    expand_agent = Use_Agent(opt, TODO='Expand diffusion prompts for me'), 
                    preloaded_model = preloaded_model
                )

    diffusion_pil = generate_example(
                            opt, new_noun, expand_agent = expand_agent, 
                            ori_img = img_pil, cond_mask = mask_1, 
                            preloaded_example_generator = preloaded_model['preloaded_example_generator'] if preloaded_model is not None else None
                        )
    # TODO: add conditional condition to diffusion via ControlNet
    _, mask_2, _ = query_middleware(
                            opt, diffusion_pil, new_noun, 
                            preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                        )
    
    if preloaded_model is None:
        sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
        sam.to(device=opt.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        mask_generator = preloaded_model['preloaded_sam_generator']['mask_generator']
    
    mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)), \
                                                                key=(lambda x: x['area']), reverse=True)
    # print(f'mask_box_list[0].keys() = {mask_box_list[0].keys()}')
    sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list]
    box_1 = match_sam_box(mask_1, sam_seg_list) # old noun
    bbox_list = [match_sam_box(x['mask'], sam_seg_list) for x in panoptic_dict]
    print(f'box_1 = {box_1}')
    print(f'bbox_list = {bbox_list}')
       
    box_name_list = [{
        'name': panoptic_dict[i]['name'],
        'bbox': bbox_list[i]
    } for i in range(len(bbox_list))]
    box_name_list.append({
        'name': old_noun,
        'bbox': box_1
    })
    
    diffusion_mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(diffusion_pil), cv2.COLOR_RGB2BGR)), \
                                                                         key=(lambda x: x['area']), reverse=True)
    box_2 = match_sam_box(mask_2, [(u['bbox'], u['segmentation'], u['area']) for u in diffusion_mask_box_list])
    
    question = Label().get_str_rescale(old_noun, new_noun, box_name_list)
    print(f'question: {question}')
    ans = get_response(edit_agent, question)
    print(f'ans: {ans}')
    
    box_0 = ans.split('[')[-1].split(']')[0]
    punctuation = re.compile('[^\w\s]+')
    box_0 = re.split(punctuation, box_0)
    box_0 = [x.strip() for x in box_0 if x!= ' ' and x!='']
    
    new_noun, x, y, w, h = box_0[0], box_0[1], box_0[2], box_0[3], box_0[4]
    print(f'new_noun, x, y, w, h = {new_noun}, {x}, {y}, {w}, {h}')
    box_0 = (int(x), int(y), int(int(w) * opt.expand_scale), int(int(h) * opt.expand_scale))

    box_0 = fix_box(box_0, (opt.H,opt.W,3))
    print(f'fixed box: (x,y,w,h) = {box_0}')
    target_mask = refactor_mask(box_2, mask_2, box_0, type=='replace')
    # 1 * h * w 
    target_mask[target_mask >= 0.5] = 0.95 if opt.mask_ablation else 1.
    target_mask[target_mask < 0.5] = 0.05 if opt.mask_ablation else 0.
    if len(target_mask.shape) > 3:
        target_mask = target_mask.unsqueeze(0)
    print('target_mask.shape = ', target_mask.shape)
    if torch.max(target_mask) <= 1. + 1e-5:
        target_mask = 255. * target_mask
        # print('plus')
    print('target_mask.shape = ', target_mask.shape)
    
    assert os.path.exists(f'./{opt.mask_dir}'), 'where is \'Semantic\' folder????'
    
    name_ = f'./{opt.mask_dir}/target_mask'
    t = 0
    while os.path.isfile(f'./{name_}-{t}.jpg'): t += 1
    cv2.imwrite(f'{name_}-{t}.jpg', tensor2img(
            repeat(target_mask, '1 ... -> c ...', c=3).clone().detach().cpu())
        )

    print(f'target_mask for replacement saved at \'{name_}-{t}.jpg\'')
    # SAVE_MASK_TEST
    print(f'target_mask.shape = {target_mask.shape}, ref_img.size = {np.array(diffusion_pil).shape}, base_img.shape = {np.array(rm_img).shape}')
    output_path, results = paint_by_example(
                                    opt, mask = target_mask, ref_img = diffusion_pil, base_img = rm_img,
                                    preloaded_example_painter = preloaded_model['preloaded_example_painter'] if preloaded_model is not None else None
                                )
    # mask required: 1 * h * w
    results = tensor2img(results)
    
    t = 0
    while os.path.isfile(f'./{output_path}-{t}.jpg'): t += 1
    cv2.imwrite(f'./{output_path}-{t}.jpg', results)
    print(f'replace result image saved at \'./{output_path}-{t}.jpg\'')
    
    return Image.fromarray(results)
    
