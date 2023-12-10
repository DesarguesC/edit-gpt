import numpy as np
import torch
from paint.bgutils import target_removing
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
# calculate IoU between SAM & SEEM
from seem.masks import middleware, query_middleware
from PIL import Image
from einops import repeat, rearrange

import cv2


transform = lambda x: repeat(rearrange(x, 'h w -> h w 1'), '... 1 -> ... b', b=3)

def find_box_idx(mask: np.array, box_list: list[tuple]):
    # print(f'mask.shape = {mask.shape}')
    cdot = [np.sum(u[1] * mask) for u in box_list]
    return np.argmax(np.array(cdot))


        

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


def Remove_Me(opt, target_noun, remove_mask=False, mask=None, resize=True):

    img_pil = Image.open(opt.in_dir).convert('RGB')
    img_pil = img_pil.resize((opt.W, opt.H))
    target_mask = None
    if remove_mask and (mask is None):
        res, target_mask, _ = query_middleware(opt, img_pil, target_noun)
        cv2.imwrite(f'./static-inpaint/res-{opt.out_name}', cv2.cvtColor(np.uint8(res), cv2.COLOR_RGB2BGR))
    elif mask is not None:
        target_mask = mask
    removed_pil = target_removing(opt=opt, target_noun=target_noun, image=img_pil,
                                  ori_shape=img_pil.size, remove_mask=remove_mask, mask=target_mask if remove_mask else None)
    removed_np = np.array(removed_pil)
    
    # TODO: use part of rm_image, cropped in bbox, to cover the original image
    
    """
        removed_np = img_np * (1. - img_mask) + removed_np * img_mask # probably not use mask at this step
        TODO: using mask to avoid the unnecessary editing of the image but failed
        Ablation: SEEM / SAM never perfectly fit the edge, which means mask hardly cover the whole object.
    """
    cv2.imwrite(f'./static-inpaint/{opt.out_name}', cv2.cvtColor(np.uint8(removed_np), cv2.COLOR_RGB2BGR))
    if remove_mask:
        return removed_np, target_mask, f'./static-inpaint/{opt.out_name}'
    else: return removed_np, f'./static-inpaint/{opt.out_name}'


def Remove_Me_SEEM(opt, target_noun, mask_generator=None, label_done=None):
    img_np, img_dragged, img_obj, img_mask, img_pil, label_done = remove_target(opt=opt,
            target_noun=target_noun, mask_generator=mask_generator, task=['Text', 'Panoptic'], label_done=label_done)

    removed_pil = target_removing(opt=opt, target_noun=target_noun, image=img_pil, ori_shape=img_pil.size)
    removed_np = np.array(removed_pil)
    print(f'removed_np.shape = {removed_np.shape}, img_mask.shape = {img_mask.shape}')

    cv2.imwrite(f'./static-inpaint/{opt.out_name}', cv2.cvtColor(np.uint8(removed_np), cv2.COLOR_RGB2BGR))

    return removed_np, label_done
