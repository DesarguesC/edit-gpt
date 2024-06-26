import re, torch, os, cv2, time
from prompt.guide import *
from prompt.util import get_image_from_box as get_img
from prompt.item import Label, get_replace_tuple, get_add_tuple
import numpy as np
from PIL import Image
from einops import repeat, rearrange
import pandas as pd
from prompt.arguments import get_args
from operations import Remove_Me, Remove_Me_lama, replace_target, create_location, Add_Object

opt = get_args()
dd = list(pd.read_csv('./key.csv')['key'])
assert len(dd) == 1
api_key = dd[0]
net_proxy = 'http://127.0.0.1:7890'
# engine='gpt-3.5-turbo-0613'
# engine='gpt-3.5-turbo'
engine = 'gpt-3.5-turbo-16k-0613'

assert os.path.exists(opt.in_dir), f'File Not Exists: {opt.in_dir}'
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(opt.out_dir):
    os.mkdir(opt.out_dir)
base_cnt = len(os.listdir(opt.out_dir))
"""
base_dir:
    -> semantic (must been done)
    -> remove (if done)
    -> replace (if done)
    -> locate (if done)
    -> add (if done)
"""

noun_list = []
label_done = Label()
class_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_sort, proxy=net_proxy)
a1 = get_response(class_agent, first_ask_sort)

sorted_class = get_response(class_agent, opt.edit_txt)
print(f'sorted class: <{sorted_class}>')
del class_agent

prompt_list = []
location = str(label_done)
edit_his = []
TURN = lambda u, image: Image.fromarray(np.uint8(get_img(image * repeat(rearrange(u[1], 'h w -> h w 1'), '... 1 -> ... c', c=3), u[0])))

if 'remove' in sorted_class:
    folder_name = f'REMOVE-{base_cnt:06}'
    setattr(opt, 'out_name', 'removed.jpg')
elif 'replace' in sorted_class:
    folder_name = f'REPLACE-{base_cnt:06}'
    setattr(opt, 'out_name', 'replaced.jpg')
elif 'locate' in sorted_class:
    folder_name = f'LOCATE-{base_cnt:06}'
    setattr(opt, 'out_name', 'located.jpg')
else:
    folder_name = f'ADD-{base_cnt:06}'
    setattr(opt, 'out_name', 'added')
    
opt.base_folder = folder_name

base_dir = os.path.join(opt.out_dir, folder_name)
opt.base_dir = base_dir
os.mkdir(base_dir)
mask_dir = os.path.join(base_dir, 'Mask')
opt.mask_dir = mask_dir
if not os.path.exists(opt.mask_dir): os.mkdir(opt.mask_dir)
print(f'base_dir: {base_dir}')
# 去掉opt.out_name这个参数

if 'remove' in sorted_class:
    # find the target -> remove -> recover the scenery
    noun_remove_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_remove, proxy=net_proxy)
    a = get_response(noun_remove_agent, remove_first_ask)
    print(a)
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    print(f'target_noun: {target_noun}')
    
    _ = Remove_Me_lama(opt, target_noun, dilate_kernel_size=opt.dilate_kernel_size) if opt.use_lama \
                        else Remove_Me(opt, target_noun, remove_mask=True)
    # TODO: recover the scenery for img_dragged in mask
    # image loaded via method Image.fromarray is 'RGB' mode by default
    print('exit from remove')
    exit(0)

elif 'replace' in sorted_class:
    # find the target -> remove -> recover the scenery -> add the new
    noun_replace_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_replace, proxy=net_proxy)
    a = get_response(noun_replace_agent, replace_first_ask)
    print(a)
    replace_tuple = get_response(noun_replace_agent, opt.edit_txt)
    print(f'replace_tuple = {replace_tuple}')
    old_noun, new_noun = get_replace_tuple(replace_tuple)
    print(f'old_noun = {old_noun}, new_noun = {new_noun}')
    """
    Remove the <replace_target>
    """
    # TODO: replace has no need of an agent; original mask and box is necessary!
    rescale_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_rescale, proxy=net_proxy)
    yes = get_response(rescale_agent, rescale_first_ask)
    print(f'rescale_agent first answers: {yes}')
    diffusion_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_expand, proxy=net_proxy)
    yes = get_response(diffusion_agent, first_ask_expand(2))
    print(f'diffusion_agent first answers: {yes}')
    replace_target(opt, old_noun, new_noun, edit_agent=rescale_agent, expand_agent=diffusion_agent)

elif 'locate' in sorted_class:
    # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
    locate_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_locate, proxy=net_proxy)
    yes = get_response(locate_agent, locate_first_ask)
    print(f'locate_agent first ask: {yes}')
    
    noun_remove_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_noun, proxy=net_proxy)
    a = get_response(noun_remove_agent, first_ask_noun)
    print(f'noun_agent first ask: {a}')
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    print(f'target_noun: {target_noun}')

    del noun_remove_agent
    create_location(opt, target_noun, edit_agent=locate_agent)

elif 'add' in sorted_class:
    add_prompt_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_addHelp, proxy=net_proxy)
    a = get_response(add_prompt_agent, addHelp_first_ask)
    print(f'add_prompt_agent first ask: {a}')
    ans = get_response(add_prompt_agent, opt.edit_txt)
    print(f'tuple_ans: {ans}')
    name, num, place = get_add_tuple(ans)
    del add_prompt_agent
    print(f'name = {name}, num = {num}, place = {place}')

    if '<NULL>' in place:
        arrange_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_add, proxy=net_proxy)
        a = get_response(arrange_agent, add_first_ask)
    else:
        arrange_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_addArrange, proxy=net_proxy)
        a = get_response(arrange_agent, addArrange_first_ask)
    print(f'arrange_agent first ask: {a}')

    diffusion_agent = get_bot(engine=engine, api_key=api_key, system_prompt=system_prompt_expand, proxy=net_proxy)
    yes = get_response(diffusion_agent, first_ask_expand(2))
    print(f'diffusion_agent first answers: {yes}')

    Add_Object(opt, name, num, place, edit_agent=arrange_agent, expand_agent=diffusion_agent)

else:
    # '<null>' in sorted_class:
    print('exit from <null>')
    exit(0)

    
    
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
    

print(f'ori: \n', label_done)
print(f'final: \n', location)

print(edit_his)
