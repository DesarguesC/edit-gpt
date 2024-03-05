import numpy as np
import torch, cv2, os
from paint.bgutils import target_removing
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
# calculate IoU between SAM & SEEM
from seem.masks import middleware, query_middleware
from PIL import Image
from einops import repeat, rearrange
from paint.bgutils import refactor_mask, match_sam_box
from paint.utils import (recover_size, resize_and_pad, load_img_to_array, save_array_to_img, dilate_mask)
from operations.utils import inpaint_img_with_lama


transform = lambda x: repeat(rearrange(x, 'h w -> h w 1'), '... 1 -> ... b', b=3)

def find_box_idx(mask: np.array, box_list: list[tuple]):
    # print(f'mask.shape = {mask.shape}')
    cdot = [np.sum(u[1] * mask) for u in box_list]
    return np.argmax(np.array(cdot))
   
def box_replace(ori_img, rm_img, target_box):
    # cv2.imwrite(f'./static-inpaint/directRM.jpg', cv2.cvtColor(np.uint8(rm_img), cv2.COLOR_RGB2BGR))
    assert ori_img.shape == rm_img.shape, f'ori_img.shape = {ori_img.shape}, rm_img.shape = {rm_img.shape}'
    print(f'ori_img.shape = {ori_img.shape}, rm_img.shape = {rm_img.shape}')
    assert ori_img.shape[-1] in [3,4] and rm_img.shape[-1] in [3,4]
    x, y, w, h = target_box
    print(f'x, y, w, h = {target_box}')
    # zeros = np.zeros_like(ori_img)
    ori_img[y:y+h, x:x+w, :] = rm_img[y:y+h, x:x+w, :]
    return ori_img
    
def remove_target(opt, target_noun, tasks=['Text'], mask_generator=None):
    assert mask_generator != None, 'mask_generator not initialized.'
    img = Image.open(opt.in_dir).convert('RGB')
    img = img.resize((ab64(img.size[0]), ab64(img.size[1])))
    # for debug
    sam_masks = mask_generator.generate(np.array(img))
    print(f'sam pieces num: {len(sam_masks)}')

    res, seem_masks, seem_box = query_middleware(opt, img, target_noun)
    print(f'type(seem_masks) = {type(seem_masks)}, seem_masks.shape = {seem_masks.shape}')
    img = img.resize((ab64(res.shape[1]), ab64(res.shape[0])))
    # res = cv2.resize(res, img.size)
    img.save('./tmp/img_np.jpg')
    img_np = np.array(img)
    
    assert img_np.shape == res.shape, f'res.shape = {res.shape}, img_np.shape = {img_np.shape}'
    sam_masks = mask_generator.generate(img_np)

    box_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
    print(f'sam pieces num: {len(sam_masks)}')

    img_idx = find_box_idx(seem_masks, box_list)
    true_mask = box_list[img_idx][1]

    print(f'true_mask.shape = {true_mask.shape}')
    mask = transform(true_mask)
    img_dragged, img_obj = img_np * mask, img_np * (1. - mask)
    return img_np, np.uint8(img_dragged), np.uint8(img_obj), mask, img

def Remove_Me_crfill(opt, target_noun, mask_generator=None, label_done=None):
    img_np, img_dragged, img_obj, img_mask, img_pil, label_done = remove_target(opt, target_noun, mask_generator, label_done)
    print(img_dragged.shape, img_obj.shape)

    img_dragged, img_obj = Image.fromarray(np.uint8(img_dragged)), Image.fromarray(np.uint8(img_obj))
    img_dragged_, img_obj_ = Image.fromarray(np.uint8(img_np * img_mask)), Image.fromarray(np.uint8(img_np * (1.-img_mask)))

    img_dragged.save('./tmp/test_out/dragged.jpg')
    img_obj.save('./tmp/test_out/obj.jpg')
    img_dragged_.save('./tmp/test_out/dragged_.jpg')
    img_obj_.save('./tmp/test_out/obj_.jpg')

    removed_pil = process_image_via_crfill(img_np, img_mask, opt) # automatically getting model
    removed_pil.save(f'static-crfill/{opt.out_name}')

    return np.array(removed_pil), label_done

def Remove_Me(opt, target_noun, remove_mask=False, replace_box=False, resize=True):
    print('-'*9 + 'Removing via Inst-Inpaint' + '-'*9)
    
    img_pil = Image.open(opt.in_dir).convert('RGB')
    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))
    
    target_mask = None
    res, target_mask, _ = query_middleware(opt, img_pil, target_noun)
    cv2.imwrite(f'./static-inpaint/res-{opt.out_name}', cv2.cvtColor(np.uint8(res), cv2.COLOR_RGB2BGR))
    print(f'seg result image saved at \'./static-inpaint/res-{opt.out_name}\'')
    
    removed_pil = target_removing(opt=opt, target_noun=target_noun, image=img_pil,
                                  ori_shape=img_pil.size, mask=target_mask if remove_mask else None)
    removed_np = np.array(removed_pil)
    
    # TODO: use part of rm_image, cropped in bbox, to cover the original image
    """
        removed_np = img_np * (1. - img_mask) + removed_np * img_mask # probably not use mask at this step
        TODO: using mask to avoid the unnecessary editing of the image but failed
        Ablation: SEEM / SAM never perfectly fit the edge, which means mask hardly cover the whole object.
    """
    box_ = (0,0,removed_np.shape[1],removed_np.shape[0])
    
    if replace_box:
        # global removed_np
        assert target_mask is not None, 'None target_mask'
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
        sam.to(device=opt.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        mask_box_list = mask_generator.generate(np.array(img_pil))
        mask_box_list = sorted(mask_box_list, key=(lambda x: x['area']), reverse=True)
        # print(f'mask_box_list[0].keys() = {mask_box_list[0].keys()}')
        # print(target_mask)
        box_ = match_sam_box(target_mask, [(u['bbox'], u['segmentation'], u['area']) for u in mask_box_list])
        # print(f'box_ = {box_}')
    
    removed_np = box_replace(np.array(img_pil), removed_np, box_)
    removed_np = cv2.cvtColor(np.uint8(removed_np), cv2.COLOR_RGB2BGR)
        
    opt.out_name = (opt.out_name + '.jpg') if not opt.out_name.endswith('.jpg') else opt.out_name
    cv2.imwrite(f'./static-inpaint/RM-{opt.out_name}', removed_np)
    print(f'removed image saved at \'./static-inpaint/RM-{opt.out_name}\'')
    # if remove_mask:
    return removed_np, target_mask, f'./static-inpaint/RM-{opt.out_name}'

def Remove_Me_lama(opt, target_noun, dilate_kernel_size=15):
    if not hasattr(opt, 'device'):
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print('-'*9 + 'Removing via LaMa' + '-'*9)
    img_pil = Image.open(opt.in_dir).convert('RGB')
    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))
    
    res, target_mask, _ = query_middleware(opt, img_pil, target_noun)
        
    print(f'target_mask.shape = {target_mask.shape}')
    target_mask_dilate = [dilate_mask(a_mask, dilate_kernel_size) for a_mask in target_mask]
    assert len(target_mask_dilate) == 1
    img_inpainted = inpaint_img_with_lama(
        np.array(np.uint8(img_pil)), target_mask_dilate[0], opt.lama_config, opt.lama_ckpt, device=opt.device
    )
    print(img_inpainted.shape)
    
    rm_output = os.path.join(opt.base_dir, 'removed.jpg')
    cv2.imwrite(rm_output, cv2.cvtColor(np.uint8(img_inpainted), cv2.COLOR_RGB2BGR))
    print(f'removed image saved at \'{rm_output}\'')
    
    return img_inpainted, target_mask, rm_output

def Remove_Me_SEEM(opt, target_noun, mask_generator=None, label_done=None):
    img_np, img_dragged, img_obj, img_mask, img_pil, label_done = remove_target(opt=opt,
            target_noun=target_noun, mask_generator=mask_generator, task=['Text', 'Panoptic'], label_done=label_done)

    removed_pil = target_removing(opt=opt, target_noun=target_noun, image=img_pil, ori_shape=img_pil.size)
    removed_np = np.array(removed_pil)
    print(f'removed_np.shape = {removed_np.shape}, img_mask.shape = {img_mask.shape}')

    cv2.imwrite(f'./static-inpaint/{opt.out_name}', cv2.cvtColor(np.uint8(removed_np), cv2.COLOR_BGR2RGB))

    return removed_np, label_done




