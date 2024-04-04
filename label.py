import re, torch, os, cv2, time
from prompt.guide import *
from prompt.util import get_image_from_box as get_img
from prompt.item import Label, get_replace_tuple, get_add_tuple
import numpy as np
from PIL import Image
from einops import repeat, rearrange
import pandas as pd
from prompt.arguments import create_parse_args
from operations import Remove_Me, Remove_Me_lama, replace_target, create_location, Add_Object

from task_planning import *

opt = get_arguments()
# for test, set "preload_all_models" = True is highly recommanded

api_key = opt.api_key
net_proxy = opt.net_proxy
engine = opt.engine
print(f'Using: {engine}')

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


preloaded_model = preload_all_models if opt.preload_all_models else None
preloaded_agent = preload_all_agents if opt.preload_all_models else None



if 'remove' in sorted_class:
    # find the target -> remove -> recover the scenery
    agent = Use_Agent(opt, TODO='find target to be removed', type=opt.llm_type) if preloaded_agent is None \
                    else preloaded_agent['find target to be removed']
    target_noun = get_response(agent, opt.edit_txt)
    print(f'\'{target_noun}\' will be removed')
    _ = Remove_Me_lama(
                opt, target_noun, input_pil = None,
                dilate_kernel_size=opt.dilate_kernel_size,
                preloaded_model = preloaded_model
            ) if opt.use_lama \
                        else Remove_Me(opt, target_noun, remove_mask=True)
    # TODO: recover the scenery for img_dragged in mask

elif 'replace' in sorted_class:
    # find the target -> remove -> recover the scenery -> add the new
    replace_agent = Use_Agent(opt, TODO='find target to be replaced', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['find target to be replaced']
    replace_tuple = get_response(replace_agent, opt.edit_txt)
    print(f'replace_tuple = {replace_tuple}')
    old_noun, new_noun = get_replace_tuple(replace_tuple)
    print(f'Replacement will happen: \'{old_noun}\' -> \'{new_noun}\'')
    del replace_agent

    # TODO: replace has no need of an agent; original mask and box is necessary!
    rescale_agent = Use_Agent(opt, TODO='rescale bbox for me', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['rescale bbox for me']
    diffusion_agent = Use_Agent(opt, TODO='expand diffusion prompts for me', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['expand diffusion prompts for me']
    replace_target(opt, old_noun, new_noun, edit_agent=rescale_agent, expand_agent=diffusion_agent)

elif 'locate' in sorted_class:
    # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
    move_agent = Use_Agent(opt, TODO='arrange a new bbox for me', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['arrange a new bbox for me']
    noun_agent = Use_Agent(opt, TODO='find target to be moved', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['find target to be moved']
    target_noun = get_response(noun_agent, opt.edit_txt)
    print(f'target_noun: {target_noun}')
    del noun_agent
    create_location(opt, target_noun, edit_agent=move_agent)

elif 'add' in sorted_class:
    add_agent = Use_Agent(opt, TODO='find target to be added', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['find target to be added']
    ans = get_response(add_agent, opt.edit_txt)
    print(f'tuple_ans: {ans}')
    name, num, place = get_add_tuple(ans)
    del add_agent
    print(f'name = {name}, num = {num}, place = {place}')

    arrange_agent = (Use_Agent(opt, TODO='generate a new bbox for me', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['generate a new bbox for me']) if '<NULL>' in place \
                    else (Use_Agent(opt, TODO='adjust bbox for me', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['adjust bbox for me'])
    diffusion_agent = Use_Agent(opt, TODO='expand diffusion prompts for me', type=opt.llm_type) if preloaded_agent is None\
                            else preloaded_agent['expand diffusion prompts for me']
    Add_Object(opt, name, num, place, edit_agent=arrange_agent, expand_agent=diffusion_agent)

else:
    # '<null>' in sorted_class:
    print('exit from <null>')
    exit(0)