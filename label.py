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

dd = list(pd.read_csv('./key.csv')['key'])
assert len(dd) == 1
api_key = dd[0]
net_proxy = 'http://127.0.0.1:7890'
engine='gpt-3.5-turbo'

# image_path = './assets/dog.jpg'
# instruction = 'close the dog\'s eyes, move the scene into a forest'
image_path = './assets/01.png'
instruction = 'turn her hair pink'


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
import cv2

sam_checkpoint = "../autodl-tmp/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image)



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


masks = mask_generator.generate(image)
masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
# type(masks), len(masks), masks[0].keys()
stack_masks = masks

box_list = [(box_['bbox'], box_['segmentation']) for box_ in masks]
print(f'len(box_list) = {len(box_list)}')
# bbox: list

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

noun_list = []
TURN = lambda u: Image.fromarray(np.uint8(get_img(image * repeat(rearrange(u[1], 'h w -> h w 1'), '... 1 -> ... c', c=3), u[0])))

# TURN = lambda u: Image.fromarray(np.uint8(get_img(image, u[0]))) # remove mask


with torch.no_grad():
    image_feature_list = [model.encode_image(preprocess(TURN(box)).unsqueeze(0).to(device)) for box in box_list]


label_done = Label()

for i in range(len(box_list)):
    box = box_list[i]
    TURN(box).save(f'./tmp/test-{i}.png')


cut_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_cut, proxy=net_proxy)
noun_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_noun, proxy=net_proxy)
edit_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_edit, proxy=net_proxy)

a1, a2, a3 = get_response(cut_agent, first_ask_cut), get_response(noun_agent, first_ask_noun), get_response(edit_agent, first_ask_edit)

print(a1, a2, a3)

from jieba import re
ins_cut = get_response(cut_agent, instruction)
ins_cut = re.split('[\.]', ins_cut)
print(len(ins_cut))

if ins_cut[-1] == '':
    del ins_cut[-1]


for i in range(len(ins_cut)):
    ins_i = ins_cut[i]
    print(f'edit prompt: {ins_i}')
    noun = get_response(noun_agent, ins_i)
    print(f'target noun: {noun}')
    noun_list.append(noun)
    text_feature = model.encode_text(clip.tokenize(['a/an/some ' + noun]).to(device))
    # print(image_feature_list[0]@text_feature.T*100.)
    with torch.no_grad():
        # logits_per_image = [model(fe, text)[0].softmax(dim=-1).cpu().numpy()[0] for fe in box_feature_list]
        img_idx = np.argmax(np.array([(100. * image_feature @ text_feature.T)[:, 0].softmax(dim=0).cpu() for image_feature in image_feature_list], dtype=np.float32))
        del image_feature_list[img_idx]
        label_done.add(stack_masks[img_idx]['bbox'], noun, img_idx)
        TURN((stack_masks[img_idx]['bbox'], stack_masks[img_idx]['segmentation'])).save(f'./tmp/noun-list/{noun}.png')
        del stack_masks[img_idx]

        
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
