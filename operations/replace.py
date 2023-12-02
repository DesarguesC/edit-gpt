from operations.remove import Remove_Me as RM
from seem.masks import middleware
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64
from PIL import Image
import numpy as np
# from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog as mt
import torch
from prompt.item import Label


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

def match_sam_box(mask: np.array, sam_list: list[tuple]):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    box_idx = np.argmax(np.sum([mask * sam_[1] for sam_ in sam_list]))
    bbox = sam_list[box_idx][0]
    del sam_list[box_idx]
    return bbox, sam_list


def preprocess_image2mask(opt, img: Image):
    # deal with the image removed target noun
    res_all, objects_masks_list = middleware(opt=opt, image=img, reftxt=None, tasks=[])
    res_all.save('./tmp/panoptic-seg.png')
    return res_all, objects_masks_list

    

def replace_target(opt, old_noun, new_noun, mask_generator=None, label_done=None, edit_agent=None):
    # assert mask_generator != None, 'mask_generator not initialized'
    assert edit_agent != None, 'no edit agent!'
    removed_np, _ = RM(opt, old_noun)
    removed_pil = Image.fromarray(removed_np)
    seg_res, objects_masks_list = preprocess_image2mask(opt, removed_pil)
    
    sam_masks = mask_generator.generate(np.array(seg_res))
    sam_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
    print(f'a box: {sam_list[0][0]}')
    
    for i in range(len(objects_masks_list)):
        mask_ = objects_masks_list[i]['mask']
        objects_masks_list[i]['bbox'], sam_list = match_sam_box(mask_, sam_list)
    
    """
        objects_masks_list = [
            {'name': ..., 'mask': ..., 'bbox': ...},
            ...
        ]
    
    """
    # print('exit')
    # exit(0)
    
    # TODO: <0> create [name, (x,y,w,h)] list to ask GPT-3.5 and arrange a place for [new_noun, (x,y,w,h)]
    series = Label().get_str(objects_masks_list)
    print(series)
    
    from revChatGPT.V3 import Chatbot
    


    # TODO: <1> create LIST
    # TODO: <2> apply an agent to generate [new_noun, (x,y,w,h)] ~ [mask]
    # TODO: <3> add some prompts to generate an image (restore required) for new_noun (via diffusion) and extract [mask, box] via SEEM
    # TODO: <4> rescale the mask and the box
    # TODO: <5> Paint-by-Example using the [mask, image] above

