import os, cv2
from jieba import re
from prompt.guide import get_response, Use_Agent, get_bot, planning_system_prompt, planning_system_first_ask
from prompt.util import get_image_from_box as get_img
from prompt.item import Label, get_replace_tuple, get_add_tuple
import numpy as np
from PIL import Image
from einops import repeat, rearrange
from prompt.arguments import get_arguments
from operations import Remove_Me, Remove_Me_lama, replace_target, create_location, Add_Object, Transfer_Me_ip2p

from preload_utils import preload_all_models, preload_all_agents

def gpt_mkdir(opt, Type = None):
    assert Type is not None, '<?>'
    base_cnt = opt.base_cnt
    if Type == 'remove':
        folder_name = f'REMOVE-{base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'removed')
        else: opt.out_name = 'removed'
    elif Type == 'replace':
        folder_name = f'REPLACE-{base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'replaced')
        else: opt.out_name = 'replaced'
    elif Type ==  'move':
        folder_name = f'LOCATE-{base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'located')
        else: opt.out_name = 'located'
    elif Type == 'add':
        folder_name = f'ADD-{base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'added')
        else: opt.out_name = 'added'
    elif Type == 'transfer':
        folder_name = f'TRANS-{base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'transfered')
        else: opt.out_name = 'transfered'
    else:
        folder_name = ''
        exit(-1)
    opt.base_folder = folder_name

    base_dir = os.path.join(opt.out_dir, folder_name)
    opt.base_dir = base_dir
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    mask_dir = os.path.join(base_dir, 'Mask')
    opt.mask_dir = mask_dir
    if not os.path.exists(opt.mask_dir): os.mkdir(opt.mask_dir)
    print(f'base_dir: {base_dir}')

    return opt

def Transfer_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None,
        preloaded_model = None,
        preloaded_agent = None,
        record_history = True,
        model_type='Ip2p'
    ):
    opt = gpt_mkdir(opt, Type='transfer')
    # no need of extra agents
    pil_return = Transfer_Me_ip2p(
                        opt, input_pil = input_pil, 
                        img_cfg = opt.img_cfg, 
                        txt_cfg = opt.txt_cfg, 
                        dilate_kernel_size = 15, 
                        preloaded_model = preloaded_model,
                        record_history = record_history,
                        model_type=model_type
                    )

    if record_history:
        method_history = (f'[{current_step:02}|{tot_step:02}]\tTransfer: 「editing has launched via InsrtuctPix2Pix」')
        return pil_return, method_history
    else: return pil_return

def Remove_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None,
        preloaded_model = None,
        preloaded_agent = None,
        record_history = True
    ):
    opt = gpt_mkdir(opt, Type='remove')

    agent = Use_Agent(opt, TODO='find target to be removed') if preloaded_agent is None \
                    else preloaded_agent['find target to be removed']
    target_noun = get_response(agent, opt.edit_txt)
    array_return, *_ = Remove_Me_lama(
                            opt, target_noun, input_pil = input_pil, 
                            dilate_kernel_size = opt.dilate_kernel_size,
                            preloaded_model = preloaded_model
                        ) if opt.use_lama \
                        else Remove_Me(opt, target_noun, remove_mask=True)
    # TODO: recover the scenery for img_dragged in mask

    if record_history:
        method_history = (f'[{current_step:02}|{tot_step:02}]\tRemove: 「{target_noun}」')
        return Image.fromarray(array_return), method_history
    else: return Image.fromarray(array_return)

def Replace_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None,
        preloaded_model = None,
        preloaded_agent = None,
        record_history = True
    ):
    opt = gpt_mkdir(opt, Type='replace')

    # find the target -> remove -> recover the scenery -> add the new
    replace_agent = Use_Agent(opt, TODO='find target to be replaced') if preloaded_agent is None\
                            else preloaded_agent['find target to be replaced']
    replace_tuple = get_response(replace_agent, opt.edit_txt)
    print(f'replace_tuple = {replace_tuple}')
    old_noun, new_noun = get_replace_tuple(replace_tuple)
    # TODO: replace has no need of an agent; original mask and box is necessary!
    rescale_agent = Use_Agent(opt, TODO='rescale bbox for me') if preloaded_agent is None\
                            else preloaded_agent['rescale bbox for me']
    diffusion_agent = Use_Agent(opt, TODO='expand diffusion prompts for me') if preloaded_agent is None\
                            else preloaded_agent['expand diffusion prompts for me']
    pil_return = replace_target(
                        opt, old_noun, new_noun, input_pil = input_pil, 
                        edit_agent = rescale_agent, expand_agent = diffusion_agent,
                        preloaded_model = preloaded_model
                    )

    if record_history:
        method_history = (f'[{current_step:02}|{tot_step:02}]\tReplace: 「{old_noun}」-> 「{new_noun}」')
        return pil_return, method_history
    else: return pil_return

def Move_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None,
        preloaded_model = None,
        preloaded_agent = None,
        record_history = True
    ):
    opt = gpt_mkdir(opt, Type='move')
    # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
    move_agent = Use_Agent(opt, TODO='arrange a new bbox for me') if preloaded_agent is None\
                            else preloaded_agent['arrange a new bbox for me']
    noun_agent = Use_Agent(opt, TODO='find target to be moved') if preloaded_agent is None\
                            else preloaded_agent['find target to be moved']
    target_noun = get_response(noun_agent, opt.edit_txt)
    print(f'target_noun: {target_noun}')

    pil_return = create_location(
                        opt, target_noun, 
                        input_pil = input_pil, 
                        edit_agent = move_agent,
                        preloaded_model = preloaded_model
                    )

    if record_history:
        method_history = (f'[{current_step:02}|{tot_step:02}]\tMove: 「{target_noun}」')
        return pil_return, method_history
    else: return pil_return

def Add_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None,
        preloaded_model = None,
        preloaded_agent = None,
        record_history = True
    ):
    opt = gpt_mkdir(opt, Type='add')

    add_agent = Use_Agent(opt, TODO='find target to be added') if preloaded_agent is None\
                            else preloaded_agent['find target to be added']
    ans = get_response(add_agent, opt.edit_txt)
    print(f'tuple_ans: {ans}')
    name, num, place = get_add_tuple(ans)

    print(f'name = {name}, num = {num}, place = {place}')
    arrange_agent = (Use_Agent(opt, TODO='generate a new bbox for me') if preloaded_agent is None\
                            else preloaded_agent['generate a new bbox for me']) if '<NULL>' in place \
                    else (Use_Agent(opt, TODO='adjust bbox for me') if preloaded_agent is None\
                            else preloaded_agent['adjust bbox for me'])
    diffusion_agent = Use_Agent(opt, TODO='expand diffusion prompts for me') if preloaded_agent is None\
                            else preloaded_agent['expand diffusion prompts for me']
    pil_return = Add_Object(
                        opt, name, num, place, 
                        input_pil = input_pil, 
                        edit_agent = arrange_agent, expand_agent = diffusion_agent,
                        preloaded_model = preloaded_model
                    )
    
    if record_history:
        method_history = (f'[{current_step:02}|{tot_step:02}]\tAdd: 「{name}」')
        return pil_return, method_history
    else: return pil_return

def get_planning_system_agent(opt):
    planning_system_agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=planning_system_prompt, proxy=opt.net_proxy)
    _ = get_response(planning_system_agent, planning_system_first_ask)
    return planning_system_agent

def get_plans(opt, planning_system_agent):
    # planning_system_agent = get_planning_system_agent(opt)
    response = get_response(planning_system_agent, opt.edit_txt)
    print(response)
    response = re.split(r"[;]", response)

    for i in range(len(response)):
        task_str = re.split(r"[(),]", response[i])
        task_str = [x.strip().strip(',') for x in task_str if x != " " and x != ""]
        assert len(task_str) == 2, f'len(task_str) != 2, task_str: {task_str}'
        response[i] = {"type": task_str[0].lower(), "command": task_str[1]}
    return response

def get_planns_directly(agent, prompt):
    response = [x.strip() for x in re.split(r"[;]", get_response(agent, prompt, mute_print=True)) if x != " " and x != ""]
    # print(f'response = {response}')
    for i in range(len(response)):
        task_str = re.split(r"[(),]", response[i])
        task_str = [x.strip().strip(',') for x in task_str if x != " " and x != ""]
        # assert len(task_str) == 2, f'len(task_str) != 2, task_str: {task_str}'
        if len(task_str) > 2:
            for i in range(2, len(task_str)):
                task_str[1] = task_str + f', {task_str[i]}'
        response[i] = {"type": task_str[0].lower(), "command": task_str[1]}
    return response


def main():
    opt = get_arguments()
    operation_menu = {
        "add": Add_Method,
        "remove": Remove_Method,
        "replace": Replace_Method,
        "move": Move_Method,
        "transfer": Transfer_Method
    }
        
    preloaded_models = preload_all_models(opt) if opt.preload_all_models else None
    preloaded_agents = preload_all_agents(opt) if opt.preload_all_agents else None

    planning_agent = get_planning_system_agent(opt)
    task_plannings = get_plans(opt, planning_agent) # [dict("type": ..., "command": ...)]

    planning_folder = os.path.join(opt.out_dir, 'plans')
    if not os.path.exists(planning_folder): os.mkdir(planning_folder)
    plan_step, tot_step = 1, len(task_plannings)
    img_pil = None # image will automatically be opened as PIL.Image in edit tools.
    method_history = []

    for plan_item in task_plannings:
        plan_type = plan_item['type']
        edit_tool = operation_menu[plan_type]
        opt.edit_txt = plan_item['command']

        img_pil, method_his = edit_tool(
                        opt, 
                        current_step = plan_step, 
                        tot_step = tot_step, 
                        input_pil = img_pil,
                        preloaded_model = preloaded_models, 
                        preloaded_agent = preloaded_agents
                    )

        method_history.append(method_his)
        img_pil.save(f'./{planning_folder}/plan{plan_step:02}({plan_type}).jpg')
        plan_step += 1
    
    # print isolation
    print()
    plan_step = 1
    for plan_item in task_plannings:
        plan_type = plan_item['type']
        command =plan_item['command']
        print(f'plan{plan_step:02}: [{plan_type.upper()}] {command}')
        plan_step += 1
    print()

    print("Operation History: ")

    for history in method_history:
        print('\t' + history)

if __name__ == "__main__":
    main()





    