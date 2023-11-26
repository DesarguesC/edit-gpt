from prompt.guide import *
import clip
from revChatGPT.V3 import Chatbot
from prompt.util import get_image_from_box as get_img
from prompt.item import Label
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

# image = cv2.imread(opt.in_dir)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(f'image.shape = {image.shape}')


  
img = Image.open(opt.in_dir)
img_np = np.array(img)
print(f'img.size = {img.size}')


# stack_masks = masks

# def save_mask(mask, mask_name):
    
    

def find_box_idx(mask: np.array, box_list: list[tuple], size: tuple):
    # print(f'mask.shape = {mask.shape}')
    cdot = [np.sum(u[1] * mask) for u in box_list]
    return np.argmax(np.array(cdot))

# model, preprocess = clip.load("ViT-B/32", device=opt.device)

noun_list = []
label_done = Label()

cut_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_cut, proxy=net_proxy)
noun_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_noun, proxy=net_proxy)
edit_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_edit, proxy=net_proxy)

a1 = get_response(cut_agent, first_ask_cut)
a2 = get_response(noun_agent, first_ask_noun)
a3 = get_response(edit_agent, first_ask_edit)
# print(a1, a2)
print(a1, a2, a3)

from jieba import re
ins_cut = get_response(cut_agent, opt.edit_txt)
ins_cut = re.split('[\.]', ins_cut)
print(len(ins_cut))

if ins_cut[-1] == '':
    del ins_cut[-1]

for x in ins_cut:
    if x == '':
        del x
    
transform = lambda x: repeat(rearrange(x, 'h w -> h w 1'), '... 1 -> ... b', b=3)

sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
sam.to(device=opt.device)
mask_generator = SamAutomaticMaskGenerator(sam)

prompt_list = []
location = str(label_done)
edit_his = []

TURN = lambda u, image: Image.fromarray(np.uint8(get_img(image * repeat(rearrange(u[1], 'h w -> h w 1'), '... 1 -> ... c', c=3), u[0])))
# sam_masks = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
print(len(ins_cut))
for i in range(len(ins_cut)):
    ins_i = ins_cut[i]
    print(f'edit prompt: {ins_i}')
    noun = get_response(noun_agent, ins_i)
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
    # print(f'len(box_list) = {len(box_list)}')
    # print(f'box_list[1][0] = {box_list[1][0]}')
    # print(f'box_list[1][1].shape = {box_list[1][1].shape}')
    # print(f'box_list[2][0] = {box_list[2][0]}')
    # print(f'box_list[2][1].shape = {box_list[2][1].shape}')
    # bbox: list
    for i in range(len(box_list)):
        box = box_list[i]
        TURN(box, res).save(f'./tmp/test-{i}.png')
    
    # seem_masks = transform(seem_masks)
    img_idx = find_box_idx(seem_masks, box_list, (res.shape[0], res.shape[1]))
    true_mask = box_list[img_idx][1]
    label_done.add(box_list[img_idx][0], noun, img_idx)
    
    # TURN((masks[img_idx]['bbox'], masks[img_idx]['segmentation']), res).save(f'./tmp/noun-list/{noun}.png')
    
    print(true_mask.shape)
    mask = transform(true_mask)
    
    Image.fromarray(np.uint8(mask * 255.)).save('./tmp/mask/m.jpg')
    
    img_dragged, img_obj = res * (1. - mask), res * mask
    print(img_dragged.shape, img_obj.shape)
    Image.fromarray(np.uint8(img_dragged)).save('./tmp/test_out/dragged.jpg')
    Image.fromarray(np.uint8(img_obj)).save('./tmp/test_out/obj.jpg')
    
    edit_op = "\"Instruction: " + ins_i.strip('\n') + "; Image: " + location.strip('\n') # + f"; Size: ({res.shape[0]},{res.shape[1]})"
    print('agent input: \n', edit_op)
    edited = get_response(edit_agent, edit_op)
    print('ori edited: \n', edited)
    edit_his.append(edited)
    edited = re.split('[\n]', edited)
    if edited[-1] == '': del edited[-1]
    location = edited[0].strip('\n')
    print('edited: ', location)
            

# for ins_i in ins_cut:
    
    

print(f'ori: \n', label_done)
print(f'final: \n', location)


print(edit_his)
