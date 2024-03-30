import torch, cv2, os
from torch import autocast
from torch.nn import functional as F
from PIL import Image
import numpy as np
from einops import repeat, rearrange

from operations.remove import Remove_Me, Remove_Me_lama
from operations.utils import get_reshaped_img
from seem.masks import middleware
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
from paint.example import paint_by_example
from paint.bgutils import match_sam_box, refactor_mask, fix_box, move_ref2base, mask_inside_box
from detectron2.data import MetadataCatalog as mt

from prompt.item import Label
from prompt.guide import get_response, Use_Agent
from prompt.gpt4_gen import gpt_4v_bbox_return
from jieba import re
from seem.masks import middleware, query_middleware
from ldm.inference_base import diffusion_inference, get_sd_models
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry

TURN = lambda x: repeat(rearrange(np.array(x.cpu().detach().numpy()), 'c h w -> h w c'), '... 1 -> ... c', c=3)

def re_mask(mask, dtype=torch.float32):
    mask[mask >= 0.5] = 1.
    mask[mask < 0.5] = 0.
    return torch.tensor(mask, dtype=dtype)

def get_area(ref_img, box):
    assert len(box) == 4
    # if box cooresponds to an area out of the ref image, then move back the box
    x, y, w, h = box
    return ref_img[y:y+h, x:x+w, :]
    
def fine_box(box, img_shape):
    # TODO: fine a box into ori-img area
    pass
    
def create_location(
        opt, 
        target, 
        input_pil: Image = None, 
        edit_agent = None,
        preloaded_model = None
    ):
    """
    -> preloaded_model
        Keys in need:
            preloaded_lama_remover
            preloaded_seem_detector
            preloaded_example_painter
    """
    """
    Mute GPT
    # assert edit_agent != None, 'no edit agent'
    """
    # move the target to the destination, editing via GPT (tell the bounding box)
    opt, img_pil = get_reshaped_img(opt, input_pil)
    # resize and prepare the original image
    if preloaded_model is None:
        sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
        sam.to(device=opt.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        mask_generator = preloaded_model['preloaded_sam_generator']['mask_generator']
    # prepare SAM, matched via SEEM
    if not opt.use_dilation:
        mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)), key=(lambda x: x['area']), reverse=True)
        sam_seg_list = [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list] if not opt.use_max_min else None
    else:
        sam_seg_list = None

    # remove the target, get the mask (for bbox searching via SAM)
    rm_img, target_mask, _ = Remove_Me_lama(
                                    opt, target, 
                                    input_pil = img_pil, 
                                    dilate_kernel_size = opt.dilate_kernel_size, 
                                    preloaded_model = preloaded_model
                                ) if opt.use_lama \
                                else Remove_Me(opt, target, remove_mask=True, replace_box=opt.replace_box)
    rm_img = Image.fromarray(rm_img, mode='RGB')

    res, panoptic_dict = middleware(
                            opt, rm_img, 
                            preloaded_seem_detector = preloaded_model['preloaded_seem_detector'] if preloaded_model is not None else None
                        )  # key: name, mask
    # destination: {[name, (x,y), (w,h)], ...} + edit-txt (tell GPT to find the target noun) + seg-box (as a hint) ==>  new box
    print(f'target_mask.shape = {target_mask.shape}')
    target_box = match_sam_box(target_mask, sam_seg_list, use_max_min=opt.use_max_min, use_dilation=(opt.use_dilation>0), dilation=opt.use_dilation, dilation_iter=opt.iteration_num)  # target box
    print(f'Matched via max-min: target_box = {target_box}')
    bbox_list = [match_sam_box(x['mask'], sam_seg_list, use_max_min=opt.use_max_min, use_dilation=(opt.use_dilation>0), dilation=opt.use_dilation, dilation_iter=opt.iteration_num) for x in panoptic_dict]
    print(target_box)
    print(f'bbox_list: {bbox_list}')

    box_name_list = [{
        'name': panoptic_dict[i]['name'],
        'bbox': bbox_list[i] if not opt.use_ratio else \
                (bbox_list[i][0]/opt.W, bbox_list[i][1]/opt.H, bbox_list[i][2]/opt.W, bbox_list[i][3]/opt.H)
    } for i in range(len(bbox_list))]
    target_box = target_box if not opt.use_ratio else \
        (target_box[0] / opt.W, target_box[1] / opt.H, target_box[2] / opt.W, target_box[3] / opt.H)
    box_name_list.append({
        'name': target,
        'bbox': target_box
    }) # as a hint

    question = Label().get_str_location(box_name_list, opt.edit_txt, (opt.W,opt.H), ratio_mode=opt.use_ratio) # => [name, (x,y), (w,h)]
    # question = f'Size: ({opt.W},{opt.H})\n' + question
    print(f'Question: \n{question}')

    """
    box_0 = target_box # Mute GPT 
    """
    box_0 = (0, 0, 0, 0)
    try_time = 0
    notes = '\n(Note that: Your response must not contain $(0,0)$ as bounding box! $w\neq 0, h\neq 0$. )'

    while box_0 == (0,0,0,0) or box_0[2] == 0 or box_0[3] == 0:
        if try_time > 0:
            if try_time > 6:
                box_0 = (0.1,0.2,0.5,0.6) if opt.use_ratio else (50,50,50,50)
                break
            print(f'Trying to fix... - Iter: {try_time}')
            print(f'QUESTION: \n{question}')
        box_ans = [str(target), '0.2', '0.4', '0.3', '0.27']
        """
        # Mute GPT
        # box_ans = [x.strip() for x in re.split(r'[\[\],()]',
        #             gpt_4v_bbox_return(opt.in_dir, opt.edit_txt).strip() if opt.gpt4_v \
        #             else get_response(edit_agent, question if try_time < 3 else (question + notes))
        #         ) if x not in ['', ' ']]
        # deal with the answer, procedure is the same as in replace.py
        """
        print(f'box_ans = {box_ans}')
        if len(box_ans) < 4:
            print('WARNING: string return')
            try_time += 1
            continue
        print(f'box_ans[0](i.e. target) = {box_ans[0]}')
        # box_ans[0]: target
        try:
            box_0 = (float(box_ans[1]), float(box_ans[2]), float(box_ans[3]) * opt.expand_scale, float(box_ans[4]) * opt.expand_scale)
        except Exception as err:
            print(f'err: box_ans = {box_ans}\nError: {err}')
            box_0 = (0, 0, 0, 0)
            try_time += 1
            continue

        print(f'box_0 before fixed: {box_0}')
        box_0 = fix_box(box_0, (1., 1., 3) if opt.use_ratio else (opt.W, opt.H, 3), target_box)
        print(f'box_0 after fixed = {box_0}')
        try_time += 1


    if opt.use_ratio:
        target_box = (int(target_box[0] * opt.W), int(target_box[1] * opt.H), int(target_box[2] * opt.W), int(target_box[3] * opt.H))
        box_0 = (int(box_0[0] * opt.W), int(box_0[1] * opt.H), int(box_0[2] * opt.W), int(box_0[3] * opt.H))

    # will be converted to int in 'refactor_mask'
    destination_mask = refactor_mask(target_box, target_mask, box_0) # torch.Tensor
    target_mask, destination_mask = re_mask(target_mask), re_mask(destination_mask)

    print(f'BEFORE: box_0={box_0}')
    print(f'{target_box} -> {box_0}')

    if torch.max(target_mask) <= 1.:
        target_mask[target_mask > 0.5] = 0.9 if opt.mask_ablation else 1.
        target_mask[target_mask <= 0.5] = 0.1 if opt.mask_ablation else 0.
    if not os.path.exists(opt.mask_dir): os.mkdir(opt.mask_dir)
    print(f'target_mask.shape = {target_mask.shape}, destination_mask.shape = {destination_mask.shape}')
    # target_mask: [1, h, w], destination_mask: [1, 3, h, w]

    name_ = f'./{opt.mask_dir}/destination'
    t = 0
    while os.path.isfile(f'{name_}-{t}.jpg'): t += 1
    cv2.imwrite(f'{name_}-{t}(after-edited).jpg', cv2.cvtColor(np.uint8((255. if torch.max(destination_mask) <= 1. else 1.) \
                            * rearrange(destination_mask.squeeze(0), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    print(f'destination-mask saved: \'{name_}-{t}.jpg\'') # 目的位置的mask (编辑后)

    name_ = f'./{opt.mask_dir}/target'
    t = 0
    while os.path.isfile(f'{name_}-{t}.jpg'): t += 1
    cv2.imwrite(f'{name_}-{t}(before-edited).jpg', cv2.cvtColor(np.uint8((255. if torch.max(target_mask) <= 1. else 1.) \
                            * rearrange(repeat(target_mask, '1 ... -> c ...', c=3), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    print(f'destination-mask saved: \'{name_}-{t}.jpg\'') # 原始图片的mask (编辑前)

    # TODO: Validate destination_mask content
    # d1, d2, d3 = destination_mask[0].chunk(3)
    # print(d1==d2)
    # print(d2==d3)
    
    img_np = np.array(img_pil)
    Ref_Image = get_area(img_np * TURN(target_mask), target_box)

    print(f'img_np.shape = {img_np.shape}, Ref_Image.shape = {Ref_Image.shape}')
    # SAVE_TEST
    print(f'Ref_Image.shape = {Ref_Image.shape}, target_mask.shape = {target_mask.shape}')

    destination_mask_sq = destination_mask[0, 0, :, :]
    # [1, 3, h, w] -> [h, w]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opt.use_dilation, opt.use_dilation))
    dilated_mask = cv2.dilate(np.uint8(destination_mask_sq * 255), kernel, iterations=opt.iteration_num)
    # dilated_mask = mask_inside_box(target_box, destination_mask_sq, kernel, opt.iteration_num)  # create from target_box
    eroded_mask = cv2.erode(np.uint8(destination_mask_sq * 255), kernel, iterations=opt.iteration_num)
    example_area = re_mask(dilated_mask / 255. * (1. - eroded_mask / 255.)) # [h w]
    example_area = repeat(torch.tensor(cv2.erode(np.uint8(example_area*255), kernel, iterations=opt.iteration_num*2)).unsqueeze(-1), 'h w 1 -> h w c', c=3) # [h w 3]
    print(f'box_0 = {box_0}')
    ref = Image.fromarray(np.uint8(Ref_Image))
    target_content = Image.fromarray(np.uint8(np.array(rm_img) * np.array(example_area)), mode='RGB')
    moved_img, moved_img_np, mask_temp = move_ref2base(box_0, ref, rm_img, mask=destination_mask)

    example_area = rearrange(torch.tensor(example_area)[:,:,0:1], 'h w c -> c h w').unsqueeze(0) # [1 1 h w]

    op_output, x_sample_ddim = paint_by_example(
                                    opt, example_area, target_content, moved_img, # i.e. img base
                                    preloaded_example_painter = preloaded_model['preloaded_example_painter'] if preloaded_model is not None else None
                                )
    print(f'x_sample_ddim.shape = {x_sample_ddim.shape}, TURN(target_mask).shape = {TURN(target_mask).shape}, img_np.shape = {img_np.shape}')

    op_output = op_output if op_output.endswith('.jpg') else (op_output + '.jpg')
    x_sample_ddim = tensor2img(x_sample_ddim.detach().cpu() * destination_mask) + np.uint8(cv2.cvtColor(moved_img_np, cv2.COLOR_BGR2RGB) * (1 - mask_temp))
    cv2.imwrite(op_output, x_sample_ddim)
    print(f'locate result image saved at \'{op_output}\'')
    
    return Image.fromarray(cv2.cvtColor(x_sample_ddim, cv2.COLOR_RGB2BGR))


