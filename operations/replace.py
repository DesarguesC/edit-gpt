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
from prompt.guide import get_response, get_bot, system_prompt_expand
from jieba import re
from seem.masks import middleware, query_middleware
from ldm.inference_base import *
from basicsr.utils import tensor2img, img2tensor
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
    res_all, objects_masks_list = middleware(opt=opt, image=img, diffusion_image=diffusion_img)
    res_all.save('./tmp/panoptic-seg.png')
    return res_all, objects_masks_list

def generate_example(opt, new_noun, expand_agent=None, use_adapter=False, ori_img: Image = None, depth_mask=None) -> Image:
    sd_model, sd_sampler = get_sd_models(opt)
    # ori_img: for depth condition generating
    adapter_features, append_to_context = None, None
    if use_adapter:
        print('-'*9 + 'Generating via Adapter' + '-'*9)
        # load adapter model to create adapter_features & append_to_context
        adapter, cond_model = get_adapter(opt), get_cond_model(opt)
        depth_cond = process_depth_cond(opt, ori_img, cond_model)
        print(f'depth_cond.shape = {depth_cond.shape}, depth_mask.shape = {depth_mask.shape}')
        depth_mask = torch.cat([torch.from_numpy(depth_mask)]*3, dim=0).unsqueeze(0).to(opt.device)
        # depth_mask = repeat(torch.from_numpy(depth_mask), 'b h w -> b c h w', c=3).to(opt.device)
        if depth_mask is not None:
            depth_mask[depth_mask < 0.5] = 0.05
            depth_mask[depth_mask >= 0.5] = 0.95
            depth_cond = depth_cond * (depth_mask * 0.8) # 1 - depth_mask ?
        cv2.imwrite(f'./static/depth_cond.jpg', tensor2img(depth_cond))
        adapter_features, append_to_context = get_adapter_feature(depth_cond, adapter)
    if isinstance(adapter_features, list):
        print(len(adapter_features))
    else:
        print(adapter_features)
    print(append_to_context)
    
    with torch.inference_mode(),  sd_model.ema_scope(), autocast('cuda'):
        seed_everything(opt.seed)
        diffusion_image = diffusion_inference(opt, new_noun, sd_model, sd_sampler, adapter_features=adapter_features, \
                                              append_to_context=append_to_context, prompt_expand_bot=expand_agent)
        diffusion_image = tensor2img(diffusion_image)
        cv2.imwrite('./static/ref.jpg', diffusion_image)
        print(f'example saved at \'./static/ref.jpg\'')
        diffusion_image = cv2.cvtColor(diffusion_image, cv2.COLOR_BGR2RGB)
    del sd_model, sd_sampler
    return Image.fromarray(np.uint8(diffusion_image)).convert('RGB') # pil


def replace_target(opt, old_noun, new_noun, edit_agent=None, expand_agent=None, replace_box=False):
    # assert mask_generator != None, 'mask_generator not initialized'
    assert edit_agent != None, 'no edit agent!'
    img_pil = Image.open(opt.in_dir).convert('RGB')
    
    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))
    
    rm_img, mask_1, _ = RM(opt, old_noun, remove_mask=True, replace_box=replace_box)
    rm_img = Image.fromarray(cv2.cvtColor(rm_img, cv2.COLOR_RGB2BGR)).convert('RGB')
    res, panoptic_dict = middleware(opt, rm_img) # key: name, mask

    diffusion_pil = generate_example(opt, new_noun, expand_agent=expand_agent, use_adapter=opt.use_adapter, ori_img=img_pil, depth_mask=mask_1)
    # TODO: add conditional condition to diffusion via ControlNet
    _, mask_2, _ = query_middleware(opt, diffusion_pil, new_noun)
    
    sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
    sam.to(device=opt.device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    mask_box_list = sorted(mask_generator.generate(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)), \
                                                                key=(lambda x: x['area']), reverse=True)
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
    box_0 = (int(x), int(y), int(w), int(h))
    target_mask = refactor_mask(box_2, mask_2, box_0)
    
    target_mask[target_mask >= 0.5] = 0.95
    target_mask[target_mask < 0.5] = 0.05
    
    
    print(f'target_mask.shape = {target_mask.shape}')
    
    cv2.imwrite(f'./outputs/mask-{opt.out_name}', cv2.cvtColor(np.uint8(255. * \
                                        rearrange(repeat(target_mask.squeeze(0), '1 ... -> c ...', c=3), 'c h w -> h w c')), cv2.COLOR_BGR2RGB))
    # SAVE_MASK_TEST
    
    output_path, results = paint_by_example(opt, mask=target_mask, ref_img=diffusion_pil, base_img=rm_img)
    # results = rearrange(results.cpu().detach().numpy(), '1 c h w -> h w c')
    # print(f'results.shape = {results.shape}')
    cv2.imwrite(output_path, tensor2img(results)) # cv2.cvtColor(np.uint8(results), cv2.COLOR_RGB2BGR)
    print('exit from replace')
    exit(0)
    
