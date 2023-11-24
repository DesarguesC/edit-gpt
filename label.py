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

image = cv2.imread(opt.in_dir)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
sam.to(device=opt.device)
mask_generator = SamAutomaticMaskGenerator(sam)
  
img = Image.open(opt.in_dir)
img_np = np.array(img)

sam_masks = mask_generator.generate(image)
sam_masks = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
# stack_masks = masks

box_list = [(box_['bbox'], box_['segmentation']) for box_ in sam_masks]
print(f'len(box_list) = {len(box_list)}')
# bbox: list

def find_box_idx(mask: np.array, box_list: list[tuple], size: tuple):
    # print(mask.shape)
    # print(size)
    def expand(in_tuple):
        # x[0-1]: box, mask; x[2-3]: image size (w h)
        x, y, w, h = in_tuple[0]
        # print((x,y,w,h))
        t1 = np.zeros_like(size)
        # print(in_tuple[1].shape)
        # print(t1[x:x+w][y:y+h].shape)
        t1[x:x+w][y:y+h] = in_tuple[1] # mask
        return t1
    cdot = [np.sum(expand(u) * mask) for u in box_list]
    return np.argmax(np.array(cdot))

# model, preprocess = clip.load("ViT-B/32", device=opt.device)

noun_list = []
TURN = lambda u: Image.fromarray(np.uint8(get_img(image * repeat(rearrange(u[1], 'h w -> h w 1'), '... 1 -> ... c', c=3), u[0])))



label_done = Label()

for i in range(len(box_list)):
    box = box_list[i]
    TURN(box).save(f'./tmp/test-{i}.png')


cut_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_cut, proxy=net_proxy)
noun_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_noun, proxy=net_proxy)
# edit_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_edit, proxy=net_proxy)

a1 = get_response(cut_agent, first_ask_cut)
a2 = get_response(noun_agent, first_ask_noun)
# a3 = get_response(edit_agent, first_ask_edit)
print(a1, a2)
# print(a1, a2, a3)

from jieba import re
ins_cut = get_response(cut_agent, opt.edit_txt)
ins_cut = re.split('[\.]', ins_cut)
print(len(ins_cut))

if ins_cut[-1] == '':
    del ins_cut[-1]

for x in ins_cut:
    if x == '':
        del x
    
transform = lambda x: repeat(rearrange(x, 'c h w -> h w c'), '... 1 -> ... b', b=3)

for i in range(len(ins_cut)):
    ins_i = ins_cut[i]
    print(f'edit prompt: {ins_i}')
    noun = get_response(noun_agent, ins_i)
    print(f'target noun: {noun}')
    noun_list.append(noun)
    # TODO: ensure the location / color / ... and etc infos are included in the noun.
    _, seem_masks = middleware(opt, img, noun)
    print(f'seem_masks.shape = {seem_masks.shape}')
    
    # seem_masks = transform(seem_masks)
    img_idx = find_box_idx(seem_masks, box_list, (image.shape[1], image.shape[2]))
    true_mask = box_list[idx][1]
    label_done.add(box_list[img_idx][0], noun, imd_idx)
    
    TURN((masks[img_idx]['bbox'], masks[img_idx]['segmentation'])).save(f'./tmp/noun-list/{noun}.png')
    
        
    mask = transform(true_mask)
    img_dragged, img_obj = img_np * (1. - mask), img_np * mask

    img_dragged.save('./tmp/test_out/dragged.jpg')
    img_obj.save('./tmp/test_out/obj.jpg')    # ===> no need of SAM !
    

exit(0)
        
prompt_list = []
location = str(label_done)
edit_his = []
for ins_i in ins_cut:
    edit_op = "\"Instruction: " + ins_i.strip('\n') + "; Image: " + location.strip('\n') + f"; Size: ({image.shape[0]},{image.shape[1]})"
    print('agent input: \n', edit_op)
    edited = get_response(edit_agent, edit_op)
    print('ori edited: \n', edited)
    edit_his.append(edited)
    edited = re.split('[\n]', edited)
    if edited[-1] == '': del edited[-1]
    location = edited[0].strip('\n')
    print('edited: ', location)
    

print(f'ori: \n', label_done)
print(f'final: \n', location)


print(edit_his)
