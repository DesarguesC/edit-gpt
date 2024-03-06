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
setattr(opt, 'api_key', list(pd.read_csv('./key.csv')['key'])[0])
setattr(opt, 'net_proxy', 'http://127.0.0.1:7890')
print(f'API is now using: {opt.engine}')

assert os.path.exists(opt.in_dir), f'File Not Exists: {opt.in_dir}'
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(opt.out_dir):
    os.mkdir(opt.out_dir)
base_cnt = len(os.listdir(opt.out_dir))
TURN = lambda u, image: Image.fromarray(np.uint8(get_img(image * repeat(rearrange(u[1], 'h w -> h w 1'), '... 1 -> ... c', c=3), u[0])))

def gpt_mkdir(Opt, Type = None):
    assert Type is not None, '<?>'
    if Type == 'remove':
        folder_name = f'REMOVE-{base_cnt:06}'
        setattr(Opt, 'out_name', 'removed.jpg')
    elif Type == 'replace':
        folder_name = f'REPLACE-{base_cnt:06}'
        setattr(Opt, 'out_name', 'replaced.jpg')
    elif Type ==  'locate':
        folder_name = f'LOCATE-{base_cnt:06}'
        setattr(Opt, 'out_name', 'located.jpg')
    elif Type == 'add':
        folder_name = f'ADD-{base_cnt:06}'
        setattr(Opt, 'out_name', 'added')
    elif Type == 'transfer':
        folder_name = f'TRANS-{base_cnt:06}'
        setattr(Opt, 'out_name', 'transfered.jpg')
    Opt.base_folder = folder_name

    base_dir = os.path.join(Opt.out_dir, folder_name)
    Opt.base_dir = base_dir
    os.mkdir(base_dir)
    mask_dir = os.path.join(base_dir, 'Mask')
    Opt.mask_dir = mask_dir
    if not os.path.exists(Opt.mask_dir): os.mkdir(Opt.mask_dir)
    print(f'base_dir: {base_dir}')

    return Opt

def Transfer_Method(opt, current_step: int, tot_step: int):
    opt = gpt_mkdir(opt, 'transfer')
    

def Remove_Method(opt, current_step: int, tot_step: int):
    opt = gpt_mkdir(opt, 'remove')

    noun_remove_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_remove, proxy=opt.net_proxy)
    a = get_response(noun_remove_agent, remove_first_ask)
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    _ = Remove_Me_lama(opt, target_noun, dilate_kernel_size=opt.dilate_kernel_size) if opt.use_lama \
                        else Remove_Me(opt, target_noun, remove_mask=True)
    # TODO: recover the scenery for img_dragged in mask
    print(f'[{current_step:02}|{tot_step:02}]\tRemove: 「{target_noun}」')

def Replace_Method(opt, current_step: int, tot_step: int):
    opt = gpt_mkdir(opt, 'replace')

    # find the target -> remove -> recover the scenery -> add the new
    noun_replace_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_replace, proxy=opt.net_proxy)
    a = get_response(noun_replace_agent, replace_first_ask)
    replace_tuple = get_response(noun_replace_agent, opt.edit_txt)
    old_noun, new_noun = get_replace_tuple(replace_tuple)
    # TODO: replace has no need of an agent; original mask and box is necessary!
    rescale_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_rescale, proxy=opt.net_proxy)
    yes = get_response(rescale_agent, rescale_first_ask)
    diffusion_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_expand, proxy=opt.net_proxy)
    yes = get_response(diffusion_agent, first_ask_expand(2)) # max sentences is 2 after expanded
    
    replace_target(opt, old_noun, new_noun, edit_agent=rescale_agent, expand_agent=diffusion_agent)
    print(f'[{current_step:02}|{tot_step:02}]\tReplace: 「{old_noun}」-> 「{new_noun}」')

def Move_Method(opt, current_step: int, tot_step: int):
    opt = gpt_mkdir(opt. 'move')

    # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
    locate_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_locate, proxy=opt.net_proxy)
    yes = get_response(locate_agent, locate_first_ask)
    
    noun_remove_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_noun, proxy=opt.net_proxy)
    a = get_response(noun_remove_agent, first_ask_noun)
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    del noun_remove_agent

    create_location(opt, target_noun, edit_agent=locate_agent)
    print(f'[{current_step:02}|{tot_step:02}]\tMove: 「{target_noun}」')

def Add_Method(opt, current_step: int, tot_step: int):
    opt = gpt_mkdir(opt, 'add')

    add_prompt_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_addHelp, proxy=opt.net_proxy)
    a = get_response(add_prompt_agent, addHelp_first_ask)
    # print(f'add_prompt_agent first ask: {a}')
    ans = get_response(add_prompt_agent, opt.edit_txt)
    # print(f'tuple_ans: {ans}')
    name, num, place = get_add_tuple(ans)
    del add_prompt_agent
    # print(f'name = {name}, num = {num}, place = {place}')

    if '<NULL>' in place:
        # bug ?
        arrange_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_add, proxy=opt.net_proxy)
        a = get_response(arrange_agent, add_first_ask)
    else:
        arrange_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_addArrange, proxy=opt.net_proxy)
        a = get_response(arrange_agent, addArrange_first_ask)
    diffusion_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_expand, proxy=opt.net_proxy)
    
    yes = get_response(diffusion_agent, first_ask_expand(2)) # max sentences is 2 after expanded
    Add_Object(opt, name, num, place, edit_agent=arrange_agent, expand_agent=diffusion_agent)
    print(f'[{current_step:02}|{tot_step:02}]\tAdd: 「{name}」')


    
    