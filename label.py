from prompt.guide import *
import clip
from revChatGPT.V3 import Chatbot
from prompt.util import get_image_from_box as get_img
from prompt.item import Label, get_replace_tuple
import torch
import numpy as np
from PIL import Image
from einops import repeat, rearrange
import pandas as pd


from revChatGPT.V3 import Chatbot
from prompt.arguments import get_args

dd = list(pd.read_csv('./key.csv')['key'])
assert len(dd) == 1
api_key = dd[0]
net_proxy = 'http://127.0.0.1:7890'
engine='gpt-3.5-turbo'

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
import cv2
# calculate IoU between SAM & SEEM
from seem.masks import middleware

opt = get_args()
opt.device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open(opt.in_dir)
img_np = np.array(img)
print(f'img.size = {img.size}')

noun_list = []
label_done = Label()


def find_box_idx(mask: np.array, box_list: list[tuple], size: tuple):
    # print(f'mask.shape = {mask.shape}')
    cdot = [np.sum(u[1] * mask) for u in box_list]
    return np.argmax(np.array(cdot))


def remove_target(opt, target_noun):
    img = Image.open(opt.in_dir)
    res, seem_masks = middleware(opt, img, target_noun)
    img_np = res
    print(f'seem_masks.shape = {seem_masks.shape}')
    print(f'res.shape = {res.shape}')
    Image.fromarray(res).save('./tmp/res.jpg')
    sam_masks = mask_generator.generate(res)

    box_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
    # bbox: list
    for i in range(len(box_list)):
        box = box_list[i]
        TURN(box, res).save(f'./tmp/test-{i}.png')

    img_idx = find_box_idx(seem_masks, box_list, (res.shape[0], res.shape[1]))
    true_mask = box_list[img_idx][1]
    label_done.add(box_list[img_idx][0], noun, img_idx)

    print(true_mask.shape)
    mask = transform(true_mask)
    img_dragged, img_obj = res * (1. - mask), res * mask
    return img_np, np.uint8(img_dragged), np.uint8(img_obj)



# model, preprocess = clip.load("ViT-B/32", device=opt.device)



# cut_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_cut, proxy=net_proxy)
class_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_sort, proxy=net_proxy)



edit_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_edit, proxy=net_proxy)

a1 = get_response(class_agent, first_ask_sort)

a3 = get_response(edit_agent, first_ask_edit)
# print(a2, a3)
print(a1, a2, a3)

sorted_class = get_response(class_agent, opt.edit_txt)
print(f'sorted class: <{sorted_class}>')

# from jieba import re
# ins_cut = get_response(cut_agent, opt.edit_txt)
# ins_cut = re.split('[\.]', ins_cut)
# print(len(ins_cut))

# if ins_cut[-1] == '':
#     del ins_cut[-1]
#
# for x in ins_cut:
#     if x == '':
#         del x
    
transform = lambda x: repeat(rearrange(x, 'h w -> h w 1'), '... 1 -> ... b', b=3)

sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
sam.to(device=opt.device)
mask_generator = SamAutomaticMaskGenerator(sam)

prompt_list = []
location = str(label_done)
edit_his = []

TURN = lambda u, image: Image.fromarray(np.uint8(get_img(image * repeat(rearrange(u[1], 'h w -> h w 1'), '... 1 -> ... c', c=3), u[0])))
# sam_masks = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
# print(len(ins_cut))


if 'remove' in sorted_class:
    # find the target -> remove -> recover the scenery
    noun_remove_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_remove, proxy=net_proxy)
    a = get_response(noun_remove_agent, first_ask_noun)
    print(a)
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    print(f'target_noun: {target_noun}')
    img_np, img_dragged, img_obj = remove_target(opt, target_noun)
    print(img_dragged.shape, img_obj.shape)
    Image.fromarray(np.uint8(img_dragged)).save('./tmp/test_out/dragged.jpg')
    Image.fromarray(np.uint8(img_obj)).save('./tmp/test_out/obj.jpg')

    Recover_Scenery_For(img_dragged)
    # TODO: recover the scenery for img_dragged in mask


if 'replace' in sorted_class:
    # find the target -> remove -> recover the scenery -> add the new
    noun_replace_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_replace, proxy=net_proxy)
    a = get_response(noun_replace_agent, first_ask_noun)
    print(a)
    replace_tuple = get_response(noun_replace_agent, opt.edit_txt)
    replace_target, replace_place = get_replace_tuple(replace_tuple)
    img_np, img_dragged_target


if 'locate' in sorted_class:
    # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
    noun_locate_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_locate, proxy=net_proxy)
    a = get_response(noun_locate_agent, first_ask_noun)
    print(a)



# for i in range(len(ins_cut)):
ins_i = opt.edit_txt
print(f'edit prompt: {ins_i}')
# noun = get_response(noun_agent, ins_i)
# noun = 'zebra'
print(f'target noun: {noun}')
noun_list.append(noun)
# TODO: ensure the location / color / ... and etc infos are included in the noun.
res, seem_masks = middleware(opt, img, noun)
img_np = res
print(f'seem_masks.shape = {seem_masks.shape}')
print(f'res.shape = {res.shape}')
Image.fromarray(res).save('./tmp/res.jpg')
sam_masks = mask_generator.generate(res)

box_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
# bbox: list
for i in range(len(box_list)):
    box = box_list[i]
    TURN(box, res).save(f'./tmp/test-{i}.png')

img_idx = find_box_idx(seem_masks, box_list, (res.shape[0], res.shape[1]))
true_mask = box_list[img_idx][1]
label_done.add(box_list[img_idx][0], noun, img_idx)

print(true_mask.shape)
mask = transform(true_mask)
img_dragged, img_obj = res * (1. - mask), res * mask
print(img_dragged.shape, img_obj.shape)
Image.fromarray(np.uint8(img_dragged)).save('./tmp/test_out/dragged.jpg')
Image.fromarray(np.uint8(img_obj)).save('./tmp/test_out/obj.jpg')
    
    # edit_op = "\"Instruction: " + ins_i.strip('\n') + "; Image: " + location.strip('\n') # + f"; Size: ({res.shape[0]},{res.shape[1]})"
    # print('agent input: \n', edit_op)
    # edited = get_response(edit_agent, edit_op)
    # print('ori edited: \n', edited)
    # edit_his.append(edited)
    # edited = re.split('[\n]', edited)
    # if edited[-1] == '': del edited[-1]
    # location = edited[0].strip('\n')
    # print('edited: ', location)
            

# for ins_i in ins_cut:
    
    

print(f'ori: \n', label_done)
print(f'final: \n', location)


print(edit_his)
