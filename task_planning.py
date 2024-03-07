import os, cv2
from jieba import re
from prompt.guide import *
from prompt.util import get_image_from_box as get_img
from prompt.item import Label, get_replace_tuple, get_add_tuple
import numpy as np
from PIL import Image
from einops import repeat, rearrange
from prompt.arguments import get_arguments
from operations import Remove_Me, Remove_Me_lama, replace_target, create_location, Add_Object, Transfer_Me_ip2p

def gpt_mkdir(opt, Type = None):
    assert Type is not None, '<?>'
    if Type == 'remove':
        folder_name = f'REMOVE-{opt.base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'removed')
        else: opt.out_name = 'removed'
    elif Type == 'replace':
        folder_name = f'REPLACE-{opt.base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'replaced')
        else: opt.out_name = 'replaced'
    elif Type ==  'locate':
        folder_name = f'LOCATE-{opt.base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'located')
        else: opt.out_name = 'located'
    elif Type == 'add':
        folder_name = f'ADD-{opt.base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'added')
        else: opt.out_name = 'added'
    elif Type == 'transfer':
        folder_name = f'TRANS-{opt.base_cnt:06}'
        if not hasattr(opt, 'out_name'): setattr(opt, 'out_name', 'transfered')
        else: opt.out_name = 'transfered'
    opt.base_folder = folder_name

    base_dir = os.path.join(opt.out_dir, folder_name)
    opt.base_dir = base_dir
    os.mkdir(base_dir)
    mask_dir = os.path.join(base_dir, 'Mask')
    opt.mask_dir = mask_dir
    if not os.path.exists(opt.mask_dir): os.mkdir(opt.mask_dir)
    print(f'base_dir: {base_dir}')

    return opt

def Transfer_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None
    ):
    opt = gpt_mkdir(opt, 'transfer')
    pil_return = Transfer_Me_ip2p(opt, input_pil=input_pil, img_cfg=opt.img_cfg, txt_cfg=opt.txt_cfg, dilate_kernel_size=15)
    print(f'[{current_step:02}|{tot_step:02}]\tTransfer: 「editing has launched via InsrtuctPix2Pix」')
    return pil_return

def Remove_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None
    ):
    opt = gpt_mkdir(opt, 'remove')

    noun_remove_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_remove, proxy=opt.net_proxy)
    a = get_response(noun_remove_agent, remove_first_ask)
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    array_return, *_ = Remove_Me_lama(opt, target_noun, input_pil=input_pil, dilate_kernel_size=opt.dilate_kernel_size) if opt.use_lama \
                        else Remove_Me(opt, target_noun, remove_mask=True)
    # TODO: recover the scenery for img_dragged in mask
    print(f'[{current_step:02}|{tot_step:02}]\tRemove: 「{target_noun}」')
    return Image.fromarray(array_return)

def Replace_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None
    ):
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
    
    pil_return = replace_target(opt, old_noun, new_noun, input_pil=input_pil, edit_agent=rescale_agent, expand_agent=diffusion_agent)
    print(f'[{current_step:02}|{tot_step:02}]\tReplace: 「{old_noun}」-> 「{new_noun}」')

    return pil_return

def Move_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None
    ):
    opt = gpt_mkdir(opt, 'move')

    # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
    locate_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_locate, proxy=opt.net_proxy)
    yes = get_response(locate_agent, locate_first_ask)
    
    noun_remove_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=system_prompt_noun, proxy=opt.net_proxy)
    a = get_response(noun_remove_agent, first_ask_noun)
    target_noun = get_response(noun_remove_agent, opt.edit_txt)
    del noun_remove_agent

    pil_return = create_location(opt, target_noun, input_pil=input_pil, edit_agent=locate_agent)
    print(f'[{current_step:02}|{tot_step:02}]\tMove: 「{target_noun}」')

    return pil_return

def Add_Method(
        opt, 
        current_step: int, 
        tot_step: int, 
        input_pil: Image = None
    ):
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
    pil_return = Add_Object(opt, name, num, place, input_pil=input_pil, edit_agent=arrange_agent, expand_agent=diffusion_agent)
    print(f'[{current_step:02}|{tot_step:02}]\tAdd: 「{name}」')

    return pil_return

def get_planning_system_agent(opt):

    """
        你是一个图像编辑系统，可以根编仅有的5个编辑工具给出编辑方案。
        你有且仅有以下5类工具做编辑: 'Add', 'Remove', 'Replace', 'Move' and 'Transfer'. 
        指令解释和效果如下所述。'Add'可以增加物体，例如可以实现"添加一个苹果", "在桌上放两个泰迪熊"；'
        Remove'可以去除物体，如可以用于"去掉桌上的梨子", "一个人把扫帚拿走了"；
        'Replace'用于替换物体，如"把狮子换成老虎", "把月亮换成太阳"；'Move'用于移动物体，
        如"把咖啡从电脑的左边拿到右边"；'Transfer'用于风格迁移，如"现代主义风格", 
        "转变成文艺复兴时期的风格". 对于输入的指令，需要你根据图像整体编辑要求给出编辑工具使用方案，
        并以$(type, method)$项的形式按顺序指明每一步的任务, 
        其中"type"是5种编辑工具中的一个(i.e. Add, Remove, Replace, Move, ransfer)
        而"method"表示实现的操作，即编辑工具的作用, 注意项与项之间以以";"分隔。
        以下是两个输入输出的例子。

        INPUT: a women enters the livingroom and take the box on the desk, while a cuckoo flies into the house.
        OUTPUT: (Remove, "remove the box on the desk"); (Add, "add a cukoo in the house")

        INPUT: "The sun went down, the sky was suddenly dark, and the birds returned to their nests."
        Output: (Remove, "remove the sun"); (Transfer, "the lights are out, darkness"); (Add, "add some birds, they are flying in the sky")
    """

    planning_system_prompt = "You are an image editing system that can give editing solutions based on only 5 editing tools. "\
                             "You have and only have the following 5 types of tools for editing: 'Add', 'Remove', 'Replace', 'Move' and 'Transfer'. "\
                             "The commands are explained and their effects are described below. "\
                             "'Add' can add objects, such as \"Add an apple\", \"Put two teddy bears on the table\"; "\
                             "\'Add\' can add objects, such as \"add an apple \",\" put two teddy bears on the table \"; "\
                             "\'Remove\' can be used to remove objects, e.g. \'Remove a pear from a table\'; "\
                             "\'A person has taken the broom away\'; \'Replace\' is used to replace an object, "\
                             "such as \"replace a lion with a tiger \", \" replace the moon with the sun \"; "\
                             "\'Move\' is used to move something, as in \'move the coffee from the left side of "\
                             "the computer to the right side\'; \'Transfer\' is used for style transfer, e.g. "\
                             "\'modernist style\', \'to Renaissance style\'. For the input instructions, "\
                             "you need to give the editing tool use plan according to the overall editing requirements of the image. "\
                             "The tasks of each step are specified in order in the form of $(type, method)$item, "\
                             "where \"type\" is one of the five editing tools (i.e. Add, Remove, Replace, Move, ransfer) and \"method\" "\
                             "indicates the operation to be implemented. That is, the role of the editing tool, "\
                             "pay attention to the items between the \";\" Separate. Here are two examples of input and output. \n"\
                             "INPUT: a women enters the livingroom and take the box on the desk, while a cuckoo flies into the house. \n"\
                             "OUTPUT: (Remove, \"remove the box on the desk\");  (Add, \"add a cukoo in the house\"). \n\n"\
                             "INPUT: \"The sun went down, the sky was suddenly dark, and the birds returned to their nests. \"\n"\
                             "Output: (Remove, \"remove the sun\"); (Transfer, \"the lights are out, darkness\"); "\
                             "(Add, \"add some birds, they are flying in the sky\")\nNote that when you are giving output, "\
                             "you mustn\'t output any other character"
    
    planning_system_first_ask = "If you have understood your task, please answer \"yes\" without any other character and "\
                                "I\'ll give you the INPUT. Note that when you are giving output, you mustn\'t output any other character"

    plannning_system_agent = get_bot(engine=opt.engine, api_key=opt.api_eky, system_prompt=planning_system_prompt, proxy=opt.net_proxy)
    a = get_response(planning, planning_system_first_ask)

    return plannning_system_agent

def get_plans(opt, planning_agent):
    planning_system_agent = get_planning_system_agent(opt)
    response = re.split(r"[;]", get_response(opt.edit_txt, planning_system_agent))
    response = [x.strip() for x in response if x != " " and x != ""]
    for i in range(len(response)):
        task_str = re.split(r"[(),\"\']", response[i])
        task_str = [x.strip() for x in task_str if x != " " and x != ""]
        assert len(task_str) == 2, f'len(task_str) != 2, task_str: {task_str}'
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
    planning_agent = get_planning_system_agent(opt)
    task_plannings = get_plans(opt, planning_agent) # [dict("type": ..., "command": ...)]
    
    planning_folder = os.path.join(opt.out_dir, 'plans')
    if not os.path.isfile(planning_folder): os.mkdir(planning_folder)
    plan_step, tot_step = 1, len(task_plannings)
    img_pil = Image.open(opt.in_dir).convert('RGB') # open input image as PIL.Image

    for plan_item in task_plannings:
        plan_type = plan_item['type']
        edit_tool = operation_menu[plan_type]
        opt.edit_txt = plan_item['command']

        img_pil = edit_tool(opt.edit_txt, plan_step, tot_step, img_pil)
        img_pil.save(f'./{planning_folder}/plan{plan_step:02}({plan_type}).jpg')
        plan_step += 1
    
    # print isolation
    print()
    plan_step = 1
    for plan_item in task_plannings:
        plan_type = plan_item['type']
        command =plan_item['command']
        print(f'plan{plan_step:02}: [{plan_type.upper()}] {command}')
    print()

if __name__ == "__main__":
    main()





    