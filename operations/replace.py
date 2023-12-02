from operations.remove import Remove_Me as RM
from seem.masks import middleware
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
from PIL import Image
import numpy as np



def preprocess_image2mask(opt, img: Image):
    # deal with the image removed target noun
    res_all, pred_dict = middleware(opt=opt, image=img, reftxt=None, tasks=[])
    res_all.save('./tmp/panoptic-seg.png')
    return res_all, pred_dict

    

def replace_target(opt, old_noun, new_noun, mask_generator=None, label_done=None):
    assert mask_generator != None, 'mask_generator not initialized'
    removed_np, _ = RM(opt, old_noun)
    # res, seem_masks = middleware(opt=opt, image=img, reftxt=opt.edit_txt, tasks=['Text', 'Panopic'])

    # img_np, img_dragged, img_obj, img_mask, img_pil, label_done = rm(opt, old_noun, mask_generator, label_done)
    # print(img_dragged.shape, img_obj.shape)
    _, pred_dict = preprocess_image2mask(opt, Image.fromarray(removed_np))
    masks, boxes = pred_dict['masks'], pred_dict['boxes']
    del pred_dict
    # TODO: <0> create [name, (x,y,w,h)] list to ask GPT-3.5 and arrange a place for [new_noun, (x,y,w,h)]



    # TODO: <1> create LIST
    # TODO: <2> apply an agent to generate [new_noun, (x,y,w,h)] ~ [mask]
    # TODO: <3> add some prompts to generate an image (restore required) for new_noun (via diffusion) and extract [mask, box] via SEEM
    # TODO: <4> rescale the mask and the box
    # TODO: <5> Paint-by-Example using the [mask, image] above


    # img_dragged, img_obj = Image.fromarray(np.uint8(img_dragged)), Image.fromarray(np.uint8(img_obj))
    # img_dragged_, img_obj_ = Image.fromarray(np.uint8(img_np * img_mask)), Image.fromarray(
    #     np.uint8(img_np * (1. - img_mask)))
    # img_dragged.save('./tmp/test_out/dragged.jpg')
    # img_obj.save('./tmp/test_out/obj.jpg')
    # img_dragged_.save('./tmp/test_out/dragged_.jpg')
    # img_obj_.save('./tmp/test_out/obj_.jpg')
    #
    # removed_pil = process_image_via_crfill(img_np, img_mask, opt)  # automatically getting model
    # removed_pil.save(f'static-crfill/{opt.out_name}')
    #
    # img, mask_all = middleware(opt, removed_pil, old_noun)




