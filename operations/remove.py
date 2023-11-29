import numpy as np
import torch
from paint.bgutils import target_removing
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
# calculate IoU between SAM & SEEM
from seem.masks import middleware
from PIL import Image
from einops import repeat, rearrange

import cv2


transform = lambda x: repeat(rearrange(x, 'h w -> h w 1'), '... 1 -> ... b', b=3)

def find_box_idx(mask: np.array, box_list: list[tuple]):
    # print(f'mask.shape = {mask.shape}')
    cdot = [np.sum(u[1] * mask) for u in box_list]
    return np.argmax(np.array(cdot))


def remove_target(opt, target_noun, mask_generator=None, label_done=None):
    assert mask_generator != None, 'mask_generator not initialized.'
    img = Image.open(opt.in_dir).convert('RGB')
    img = img.resize((ab64(img.size[0]), ab64(img.size[1])))
    
    
    # for debug
    sam_masks = mask_generator.generate(np.array(img))
    print(f'sam pieces num: {len(sam_masks)}')
    
    
    res, seem_masks = middleware(opt, img, target_noun, tasks=['Text'])
    print(f'type(seem_masks) = {type(seem_masks)}, seem_masks.shape = {seem_masks.shape}')
    img = img.resize((ab64(res.shape[1]), ab64(res.shape[0])))
    # res = cv2.resize(res, img.size)
    img.save('./tmp/img_np.jpg')
    img_np = np.array(img)
    
    assert img_np.shape == res.shape, f'res.shape = {res.shape}, img_np.shape = {img_np.shape}'
    sam_masks = mask_generator.generate(img_np)

    box_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
    print(f'sam pieces num: {len(sam_masks)}')
    # bbox: list
    # for i in range(len(box_list)):
    #     box = box_list[i]
    #     TURN(box, res).save(f'./tmp/test-{i}.png')

    img_idx = find_box_idx(seem_masks, box_list)
    true_mask = box_list[img_idx][1]
    if label_done != None:
        label_done.add(box_list[img_idx][0], target_noun, img_idx)

    print(f'true_mask.shape = {true_mask.shape}')
    mask = transform(true_mask)
    img_dragged, img_obj = img_np * mask, img_np * (1. - mask)
    return img_np, np.uint8(img_dragged), np.uint8(img_obj), mask, img, label_done



def Remove_Me(opt, target_noun, mask_generator=None, label_done=None):
    input_image = Image.open(opt.in_dir)
    
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
    
    # try:
    removed_pil = target_removing(opt=opt, target_noun=target_noun, image=img_pil, ori_shape=img_pil.size)
    removed_pil.save(f'static-inpaint/{opt.out_name}')
    # except Exception as err:
        # print(err)
        # print('inst-inpaint error didn\'t be handled')
    
    return label_done

