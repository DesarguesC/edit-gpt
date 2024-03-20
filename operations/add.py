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

    if preloaded_model is None:
        # load SAM
        sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
        sam.to(device=opt.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        mask_generator = preloaded_model['preloaded_sam_generator']['mask_generator']

    mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)), key=(lambda x: x['area']), reverse=True)
    sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list]


    if '<NULL>' in place:
        # system_prompt_add -> panoptic
        _, panoptic_dict = middleware(
                                opt, img_pil,
                                preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                            )

        place_mask_list = [item['mask'] for item in panoptic_dict]
        panoptic_list = []
        
        for i in range(len(panoptic_dict)):
            panoptic_list.append({
                'name': panoptic_dict[i]['name'],
                'bbox': match_sam_box(place_mask_list[i], sam_seg_list) if not opt.use_max_min else match_sam_box(mask=place_mask_list[i]) # only mask input -> extract max-min coordinates as bounding box
            })
        question = Label().get_str_add_panoptic(panoptic_list, name, (opt.W,opt.H))
        # remained to be debug
    else:
        # system_prompt_addArrange -> name2place
        _, place_mask, _ = query_middleware(
                                    opt, img_pil, place,
                                    preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                                )
        # old mask
        place_box = match_sam_box(place_mask, sam_seg_list) if not opt.use_max_min else match_sam_box(mask=place_mask) # only mask input -> extract max-min coordinates as bounding box
        print(f'place_box = {place_box}')
        question = Label().get_str_add_place(place, name, (opt.W,opt.H), place_box)

    print(f'question: \n{question}\n num = {num}')
    
    for i in range(num):

        fixed_box = (0,0,0,0)
        try_time = 0
        notes = '\n(Note that: Your response must not contain $(0,0)$ as bounding box! $w\neq 0, h\neq 0$. )'

        while fixed_box == (0,0,0,0) or fixed_box[2] == 0 or fixed_box[3] == 0:
            if try_time > 0:
                if try_time > 6:
                    fixed_box = (50,50,50,50)
                    break
                print(f'Trying to fix... - Iter: {try_time}')
                print(f'QUESTION: \n{question}')
            box_ans = [x.strip() for x in re.split(r'[\[\],()]', 
                        gpt_4v_bbox_return(opt.in_dir, opt.edit_txt).strip() if opt.gpt4_v \
                        else get_response(edit_agent, question if try_time < 3 else (question + notes))
                    ) if x not in ['', ' ']]
            # deal with the answer, procedure is the same as in replace.py
            print(f'box_ans = {box_ans}')
            if len(box_ans) < 4:
                print('WARNING: string return')
                try_time += 1
                continue
            print(f'box_ans[0](i.e. target) = {box_ans[0]}')
            x, y, w, h = float(box_ans[1]), float(box_ans[2]), float(box_ans[3]), float(box_ans[4])
            fixed_box = (int(x), int(y), int(w * opt.expand_scale), int(h * opt.expand_scale))
            print(f'box_0 before fixed: {fixed_box}')
            fixed_box = fix_box(fixed_box, (opt.W,opt.H,3))
            print(f'box_0 after fixed = {fixed_box}')

            try_time += 1
        
        print(f'ans_box = {fixed_box}')
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
        box_example = match_sam_box(mask_example, [
            (u['bbox'], u['segmentation'], u['area']) for u in \
                    sorted(mask_generator.generate(cv2.cvtColor(np.array(diffusion_pil), cv2.COLOR_RGB2BGR)), \
                                           key=(lambda x: x['area']), reverse=True)
                    ]) if not opt.use_max_min else match_sam_box(mask=mask_example) # only mask input -> extract max-min coordinates as bounding box

        target_mask = refactor_mask(box_example, mask_example, fixed_box, use_max_min=opt.use_max_min)

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






