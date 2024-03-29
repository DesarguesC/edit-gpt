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
from prompt.guide import get_response, get_bot, system_prompt_expand
from jieba import re
from seem.masks import middleware, query_middleware
from ldm.inference_base import *
from paint.control import *
from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
from einops import repeat, rearrange
import torch, cv2, os
from paint.example import generate_example
from operations.utils import get_reshaped_img
from prompt.gpt4_gen import gpt_4v_bbox_return

def Add_Object(
        opt, 
        name: str, 
        num: int, 
        place: str, 
        input_pil: Image = None, 
        edit_agent = None, 
        expand_agent = None,
        preloaded_model = None,
    ):
    """
    -> preloaded_model
        Keys in need:
            preloaded_sam_generator
            preloaded_example_generator
            preloaded_example_painter
    """
    
    assert edit_agent != None, 'no edit agent!'
    opt, img_pil = get_reshaped_img(opt, input_pil)
    print(f'ADD: (name, num, place) = ({name}, {num}, {place})')
    assert os.path.exists(opt.base_dir), 'where is base_dir ?'
    add_path = os.path.join(opt.base_dir, 'added')
    if not os.path.exists(add_path): os.mkdir(add_path)

    if not (opt.use_dilation or opt.use_max_min):
        if preloaded_model:
            # load SAM
            sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
            sam.to(device=opt.device)
            mask_generator = SamAutomaticMaskGenerator(sam)
        else:
            mask_generator = preloaded_model['preloaded_sam_generator']['mask_generator']
        mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)), key=(lambda x: x['area']), reverse=True)
        sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list]
    else:
        sam_seg_list = None

    if '<NULL>' in place:
        # system_prompt_add -> panoptic
        _, panoptic_dict = middleware(
                                opt, img_pil,
                                preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                            )

        place_mask_list = [item['mask'] for item in panoptic_dict]
        panoptic_list = []
        
        for i in range(len(panoptic_dict)):
            temp_item = {
                'name': panoptic_dict[i]['name'],
                'bbox': match_sam_box(place_mask_list[i], sam_seg_list, use_max_min=opt.use_max_min, use_dilation=(opt.use_dilation>0), dilation=opt.use_dilation, dilation_iter=opt.dilation_iter) # only mask input -> extract max-min coordinates as bounding box
            }
            if opt.use_ratio:
                temp_item['bbox'][0], temp_item['bbox'][1] = temp_item['bbox'][0] / opt.W, temp_item['bbox'][1] / opt.H
                temp_item['bbox'][2], temp_item['bbox'][3] = temp_item['bbox'][2] / opt.W, temp_item['bbox'][3] / opt.H
            panoptic_dict.append(temp_item)

        question = Label().get_str_add_panoptic(panoptic_list, name, (opt.W,opt.H), ratio_mode=opt.use_ratio)
        # remained to be debug
    else:
        # system_prompt_addArrange -> name2place
        _, place_mask, _ = query_middleware(
                                    opt, img_pil, place,
                                    preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                                )
        # old mask
        place_box = match_sam_box(place_mask, sam_seg_list, use_max_min=opt.use_max_min, use_dilation=(opt.use_dilation>0), dilation=opt.use_dilation, dilation_iter=opt.dilation_iter) # only mask input -> extract max-min coordinates as bounding box
        if opt.use_ratio:
            place_box = (place_box[0]/opt.W, place_box[1]/opt.H, place_box[2]/opt.W, place_box[3]/opt.H)
        print(f'place_box = {place_box}')
        question = Label().get_str_add_place(place, name, (opt.W,opt.H), place_box, ratio_mode=opt.use_ratio)

    print(f'question: \n{question}\n num = {num}\n(Using Ratio = {opt.use_ratio})')
    
    for i in range(num):

        fixed_box = (0,0,0,0)
        try_time = 0
        notes = '\n(Note that: Your response must not contain $(0,0)$ as bounding box! $w\neq 0, h\neq 0$. )'

        while fixed_box == (0,0,0,0) or fixed_box[2] == 0 or fixed_box[3] == 0:
            if try_time > 0:
                if try_time > 4:
                    fixed_box = (0.25,0.25,0.1,0.1) if opt.use_ratio else (50,50,50,50)
                    break
                print(f'Trying to fix... - Iter: {try_time}')
                print(f'QUESTION: \n{question}')
            box_ans = [x.strip() for x in re.split(r'[\[\],()]', 
                        gpt_4v_bbox_return(opt.in_dir, opt.edit_txt).strip() if opt.gpt4_v \
                        else get_response(edit_agent, question if try_time < 2 else (question + notes))
                    ) if x not in ['', ' ']]
            # deal with the answer, procedure is the same as in replace.py
            print(f'box_ans = {box_ans}')
            if len(box_ans) < 4:
                print('WARNING: string return')
                try_time += 1
                continue
            print(f'box_ans[0](i.e. target) = {box_ans[0]}')
            try:
                x, y, w, h = float(box_ans[1]), float(box_ans[2]), float(box_ans[3]) * opt.expand_scale, float(box_ans[4]) * opt.expand_scale
            except Exception as err:
                print(f'err: box_ans = {box_ans}\bError: {err}')
                fixed_box = (0,0,0,0)
                try_time += 1
                continue

            fixed_box = (x, y, w, h)
            print(f'box_0 before fixed: {fixed_box}')
            fixed_box = fix_box(fixed_box, (1., 1., 3) if opt.use_ratio else (opt.W, opt.H, 3))
            print(f'box_0 after fixed = {fixed_box}')
            try_time += 1

        # fixed_box will be converted to integer before 'refactor_mask'


        # generate example
        diffusion_pil = generate_example(
                            opt, name, expand_agent = expand_agent, ori_img = img_pil, 
                            preloaded_example_generator = preloaded_model['preloaded_example_generator'] if preloaded_model is not None else None
                        )
        
        # query mask-box & rescale
        _, mask_example, _ = query_middleware(
                                    opt, diffusion_pil, name,
                                    preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                                )
        print(f'diffusion_pil.size = {diffusion_pil.size}, mask_example.shape = {mask_example.shape}')

        assert mask_example.shape[-2:] == (opt.H, opt.W), f'mask_example.shape = {mask_example.shape}, opt(H, W) = {(opt.H, opt.W)}'


        if not (opt.use_dilation or opt.use_max_min):
            mask_sam_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(diffusion_pil), cv2.COLOR_RGB2BGR)), key=(lambda x: x['area']), reverse=True)
            sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_sam_list]
        else:
            sam_seg_list = None

        # input: normal. output: normal | bounding box has been recovered from ratio space
        box_example = match_sam_box(mask_example, sam_seg_list, use_max_min=opt.use_max_min, use_dilation=(opt.use_dilation>0), dilation=opt.use_dilation, dilation_iter=opt.dilation_iter)  # only mask input -> extract max-min coordinates as bounding box

        if opt.use_ratio:
            box_example = (int(box_example[0] * opt.W), int(box_example[1] * opt.H), int(box_example[2] * opt.W), int(box_example[3] * opt.H))
            fixed_box = (int(fixed_box[0] * opt.W), int(fixed_box[1] * opt.H), int(fixed_box[2] * opt.W), int(fixed_box[3] * opt.H))
        # will be converted to int in 'refactor_mask'
        print(f'ans_box = {fixed_box}') # recovered from ratio

        target_mask = refactor_mask(box_example, mask_example, fixed_box)

        # TODO: save target_mask
        print(f'In \'add\': target_mask.shape = {target_mask.shape}, mask_example.shape = {mask_example.shape}')
        name_ = f'./{opt.mask_dir}/mask'
        t = 0
        while os.path.isfile(f'{name_}-{t}.jpg'): t += 1
        cv2.imwrite(f'{name_}-{t}.jpg', tensor2img(
            (255. if torch.max(target_mask) <= 1. else 1.) * target_mask.squeeze(0)
        ))
        print(f'target mask for adding is saved at \'{name_}-{t}.jpg\'')
        
        # paint-by-example
        x, y, w, h = box_example
        # print(f'mask_example.shape = {mask_example.shape}')
        np_img_ = np.array(diffusion_pil) * rearrange(repeat(mask_example, '1 h w -> c h w', c=3), 'c h w -> h w c')
        # print(f'np_img_.shape = {np_img_.shape}')
        diffusion_pil = Image.fromarray(np.uint8(np_img_[y:y+h, x:x+w, :]) * (255 if np.max(np_img_) <= 1. else 1)).convert('RGB')
        _, painted = paint_by_example(
                            opt, mask=target_mask, ref_img=diffusion_pil, base_img=img_pil, 
                            preloaded_example_painter = preloaded_model['preloaded_example_painter'] if preloaded_model is not None else None
                        )
        output_path = os.path.join(add_path, f'added_{i}.jpg')
        painted = tensor2img(painted)
        cv2.imwrite(output_path, painted)
        img_pil = Image.fromarray(cv2.cvtColor(painted, cv2.COLOR_BGR2RGB))
        print(f'Added: Image \'added_{i}.jpg\' saved at \'{output_path}\' folder.')

    # print('exit from add')
    # exit(0)
    return img_pil






