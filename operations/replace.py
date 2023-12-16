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
from paint.control import *
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

def generate_example(opt, new_noun, expand_agent=None, use_adapter=False, ori_img: Image = None, cond_mask=None) -> Image:
    
    if opt.use_inpaint_adpater:
        from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, \
            AutoencoderKL
        from diffusers.utils import load_image, make_image_grid
        from controlnet_aux.lineart import LineartDetector
        import torch

        # load adapter
        adapter = T2IAdapter.from_pretrained(
            "../autodl-tmp/TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16, varient="fp16", local_files_only=True
        ).to("cuda")

        # load euler_a scheduler
        model_id = '../autodl-tmp/stabilityai/stable-diffusion-xl-base-1.0'
        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True)
        vae = AutoencoderKL.from_pretrained("../autodl-tmp/madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, local_files_only=True)
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", local_files_only=True
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        line_detector = LineartDetector.from_pretrained("../autodl-tmp/lllyasviel/Annotators", local_files_only=True).to("cuda")

        # image = load_image(url)
        image = line_detector(
            ori_img, detect_resolution=384, image_resolution=1024
        )

        prompt = "Ice dragon roar, 4k photo"
        negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
        gen_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=50,
            adapter_conditioning_scale=0.8,
            guidance_scale=7.5,
        ).images[0]
        gen_images.save('static/ref.jpg')
        return gen_images # PIL.Image
    
    sd_model, sd_sampler = get_sd_models(opt)
    # ori_img: for depth condition generating
    adapter_features, append_to_context = None, None
    # if use_adapter:
    #     print('-'*9 + 'Generating via Depth Adapter' + '-'*9)
    #     # load adapter model to create adapter_features & append_to_context
    #     adapter, cond_model = get_adapter(opt), get_depth_model(opt)
    #     depth_cond = process_depth_cond(opt, ori_img, cond_model)
    #     assert cond_mask is not None
    #     # print(f'depth_cond.shape = {depth_cond.shape}, cond_mask.shape = {cond_mask.shape}')
    #     cond_mask = torch.cat([torch.from_numpy(cond_mask)]*3, dim=0).unsqueeze(0).to(opt.device)
    #     if cond_mask is not None:
    #         cond_mask[cond_mask < 0.5] = 0.05
    #         cond_mask[cond_mask >= 0.5] = 0.95
    #         # TODO: check if mask smoothing is needed
    #         depth_cond = depth_cond * (cond_mask * 0.8) # 1 - cond_mask ?
    #     cv2.imwrite(f'./static/depth_cond.jpg', tensor2img(depth_cond))
    #     adapter_features, append_to_context = get_adapter_feature(depth_cond, adapter)
    if use_adapter:
        print('-'*9 + 'Generating via Style Adapter' + '-'*9)
        adapter, cond_model = get_adapter(opt), get_style_model(opt)
        print(f'BEFORE: cond_img.size = {ori_img.size}')
        style_cond = process_style_cond(opt, ori_img, cond_model) # not a image
        print(f'style_cond.shape = {style_cond.shape}, cond_mask.shape = {cond_mask.shape}')
        # resize mask to the shape of style_cond ?
        # cond_mask = torch.cat([torch.from_numpy(cond_mask)]*3, dim=0).unsqueeze(0).to(opt.device)
        # print(f'cond_mask.shape = {cond_mask.shape}')
        # if cond_mask is not None:
            # cond_mask[cond_mask < 0.5] = 0.05
            # cond_mask[cond_mask >= 0.5] = 0.95
            # TODO: check if mask smoothing is needed
        # style_cond = style_cond * (style_mask * 0.8) # 1 - cond_mask ?
        cv2.imwrite(f'./static/style_cond.jpg', tensor2img(style_cond))
        adapter_features, append_to_context = get_adapter_feature(style_cond, adapter)
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

    diffusion_pil = generate_example(opt, new_noun, expand_agent=expand_agent, use_adapter=opt.use_inpaint_adapter, ori_img=img_pil, cond_mask=mask_1)
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
    
    output_path, results = paint_by_example(opt, mask=target_mask, ref_img=diffusion_pil, base_img=rm_img, use_adapter=opt.use_pbe_adapter)
    # results = rearrange(results.cpu().detach().numpy(), '1 c h w -> h w c')
    # print(f'results.shape = {results.shape}')
    cv2.imwrite(output_path, tensor2img(results)) # cv2.cvtColor(np.uint8(results), cv2.COLOR_RGB2BGR)
    print('exit from replace')
    exit(0)
    
