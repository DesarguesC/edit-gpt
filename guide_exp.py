import os, time, json
from prompt.guide import get_response, get_bot
from basicsr.utils import tensor2img, img2tensor
from random import randint
from PIL import Image
import numpy as np
from task_planning import Replace_Method, Move_Method
from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from detectron2.data import MetadataCatalog
from preload_utils import *

"""
    你是一个instruction生成器，你需要根据描述两幅相似图像的caption中的文字差异，生成一条能够通过“replace” 实现图像编辑的指令，例如：
    Input: 1.mountains of scotland under a bright sunny sky 2. mountains of scotland under a rainy sky
    Output: replace "bright sunny sky" with a "rainy sky".
    我们更希望你使用"replace A with B"的句型。另外，如果你认为这两条caption之间不能用一条只用“replace”方法的instruction实现，
    请输出“NULL”。例如：
    Input: 1. Aspen Country II Painting 2. Aspen Country II Cartoon
    Output: NULL
    因为这是一个风格迁移方面的变换
    注意，你的输出中禁止包含其他多余的字符。
"""

system_prompt_gen_replace_instructions = "You are an instruction generator, and you need to generate an instruction that "\
                                         "enables image editing via \"replace\" based on the textual differences in a "\
                                         "caption describing two similar images, for example: \n"\
                                         "Input: 1.mountains of scotland under a bright sunny sky 2. mountains of scotland under a rainy sky\n"\
                                         "Output: replace \"bright sunny sky\" with a \"rainy sky\". \n"\
                                         "We prefer you to use the \"replace A with B\" pattern. Also, if you think that the two captions "\
                                         "can't be separated by an instruction that only uses the \"replace\" method, just output \"NULL\". "\
                                         "For instance:\n"\
                                         "Input: 1. Aspen Country II Painting 2. Aspen Country II Cartoon\n"\
                                         "Output: NULL\nFor the fact that they are style transfering concerned instructions. "\
                                         "Note that you are forbidden to include any other extra characters in your output."


# 输入：cpation, label, bounding box
"""
    你是一个位置生成器，你需要根据输入的caption和label，为处在bounding box描述位置处的物体生成一个使用文字描述的位置。
    你获得的输入是：caption，label，（x,y,w,h）
    这里的(x,y,w,h)是bounding box，其含义为：(x,y)表示bounding box左上角的点的坐标，(w,h)为bounding box的宽度和高度。
    你需要将生成的描述性位置输出，同时给出另一个目标位置的文字描述，使得物体可以从当前位置被移动到目标位置，例如：
    Input: an apple is on the desk, apple, (100,100,50,70)
    Output: on the desk; under the desk
    Input: an apple on the desk, desk, (30,140,300,240)
    Output: desk on the left; desk on the right
    每个输出中的两个位置A和B用";"分割，你的输出禁止包含多余的无关字符
"""

system_prompt_gen_move_instructions = "You are a position generator and you need to generate a textual position "\
                                      "for an object at the position described by a bounding box, based on the input caption and label. "\
                                      "The input you get is: caption, label, (x,y,w,h). Here (x,y,w,h) is the bounding box, "\
                                      "which means: (x,y) represents the coordinates of the point in the upper left corner of "\
                                      "the bounding box, and (w,h) is the width and height of the bounding box. "\
                                      "You need to output the generated descriptive position with another textual description "\
                                      "of the target position, so that the object can be moved from its current position to the target position, for example: \n"\
                                      "Input: an apple is on the desk, apple, (100,100,50,70)\n"\
                                      "Output: on the desk; under the desk\n"\
                                      "Input: an apple on the desk, desk, (30,140,300,240)\n"\
                                      "Output: on the left; on the right\n"\
                                      "The two positions A and B in each output are separated by \";\", "\
                                      "and your output is forbidden to contain extra extraneous characters."
    

system_prompt_edit_sort =   'You are an expert in text classiffication,  and there are 5 classes in total.'\
                            '1. \"Remove\": determines whether the text removes the object, and if so, it is \"Romove\". '\
                            '2. \"Replace\": determine whether the text replaces the object, and if so, its category is \"Replace\". '\
                            '3. \"Move\": determine whether the text moves the object. If it does, the category is \"Move\". '\
                            '4. \"Add\": determine whether the text add several object. If it does, the category is \"Add\". '\
                            '5. \"Transfer\": determine whether the text is to do style transfering. If it does, the category is \"Transfer\". '\
                            'Note that the text is an editing instruction for the picture. We ensure all the text input is included in these 5 classes. \n'\
                            'For instance: \n'\
                            'Input: make the Ferris Wheel a giant hamster wheel\nOutput: \"Replace\"\n'\
                            'Input: make it an oil painting\nOutput: \"Transfer\"\n'\
                            'Input: have the ranch be a zoo\nOutput: \"Replace\"\n'\
                            'Note that you are forbidden to include any other extra characters in your output.'



def preload_replace_model(opt):
    return {
        'preloaded_example_generator': preload_example_generator(opt), 
        # XL - 8272 MiB, XL_ad - 8458 MiB, V1.5 - 10446 MiB
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt) # 10446 MiB
    }

def preload_move_model(opt):
    return {
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt) # 10446 MiB

    }

def use_exp_agent(opt, system_prompt):
    agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=system_prompt, proxy=opt.net_proxy)
    return agent

def write_replace_instruction(agent, path, caption1, caption2):
    Input = f'1. {caption1} 2. {caption2}'
    Output = get_response(agent, Input)
    with open(path, 'w') as f:
        f.write(Output)
    return Output

def write_move_instruction(agent, path, caption, label, bbox):
    Input = f'{caption}, {label}, {bbox}'
    Output = get_response(agent, Input)
    # Output = Output.split(';')
    # assert len(Output) == 2, f'Output = {Output}'
    # Output = f'{Output[0]}, {Output[1]}'
    with open(path, 'w') as f:
        f.write(Output)
    return Output

def read_original_prompt(path_to_json):
    assert path_to_json.endswith('.json')
    with open(path_to_json, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    prompt1 = data['input']
    edit = data['edit']
    prompt2 = f'{prompt1}, with {edit}'
    return (prompt1, prompt2, edit)


def Val_Replace_Method(opt):
    from prompt.arguments import get_arguments
    agent = use_exp_agent(opt, system_prompt_edit_sort)
    val_folder = '../autodl-tmp/clip-filtered/shard-00/'
    folders = os.listdir(val_folder)
    length = len(folders)
    selected_list = []
    executed_list = []
    execute_img_cnt = 0

    from preload_utils import preload_all_agents
    preloaded_replace_model =  preload_replace_model(opt) if opt.preload_all_models else None
    preloaded_agent =  preload_all_agents(opt) if opt.preload_all_agents else None

    real_fake_image_list = []
    fake_image_list = []
    caption_before_list = []
    caption_after_list = []

    # 4-6 images in a folder
    while len(executed_list) < opt.test_group_num:
        while True:
            folder = folders[randint(0, length)]
            if folder in selected_list: continue
            else:
                selected_list.append(folder)
                break
        work_folder = os.path.join(val_folder, folder)
        json_path = os.path.join(work_folder, 'prompt.json')
        c1, c2, edit = read_original_prompt(json_path)
        sorted = get_response(agent, edit)
        if not 'replace' in sorted.lower(): continue
        else: executed_list.append(folder)

        name_list = [img.split('_')[0] for img in os.listdir(work_folder) if img.endswith('.jpg')]
        name_list = list(set(name_list))
        opt.edit_txt = edit
        for name in name_list:
            img_path = os.path.join(work_folder, f'{name}_0.jpg')
            img_pil = Image.open(img_path).convert('RGB')
            output_pil = Replace_Method(opt, 0, 0, img_pil, preloaded_replace_model, preloaded_agent, record_history=False)
            caption_before_list.append(c1)
            caption_after_list.append(c2)
            fake_image_list.append(output_pil) # pil list
            real_fake_image_list.append(Image.open(os.path.join(work_folder, f'{name}_1.jpg')).convert('RGB')) # pil list
            execute_img_cnt += 1
        
        print(f'Images have been Replaced: {execute_img_cnt}')

    clip_directional_similarity = cal_similarity(real_fake_image_list, fake_image_list, caption_before_list, caption_after_list)
    print(f"clip directional similarity: {clip_directional_similarity}")
    with open("models/clip_directional_similarity_Replace.txt", "w") as f:
        f.write(str(clip_directional_similarity))

    fid_score = cal_fid(real_fake_image_list, fake_image_list)
    print(f"FID Score: {fid_score}")
    with open("models/fid_score_Replace_.txt", "w") as f:
        f.write(str(fid_score))
    
    del preloaded_agent, preloaded_replace_model
    # consider if there is need to save all images replaced

    
def Val_Move_Method(opt):
    val_folder = '../autodl-tmp/COCO/val2017/'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    from prompt.arguments import get_arguments
    agent = use_exp_agent(opt, system_prompt_gen_move_instructions)
    
    caption_before_list = captions_after_list = []
    image_before_list = image_after_list = []

    # for validation after
    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        caption = json.load(f)    
    # query caption via image_id
    captions_dict = {}

    preloaded_move_model = preload_move_model(opt) if opt.preload_all_models else None
    preloaded_agent =  preload_all_agents(opt) if opt.preload_all_agents else None

    for x in caption['annotations']:
        image_id = str(x['image_id'])
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + ', ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data = json.load(f)
    
    length = len(data['annotations'])
    selected_list = []
    
    while len(selected_list) < opt.test_group_num:
        while True:
            idx = randint(0, length)
            if idx in selected_list: continue
            else: break
        selected_list.append(idx)
        annotation = data['annotations'][idx]

        x, y, w, h = annotation['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        img_id, label_id = annotation['image_id'], annotation['category_id']
        caption = captions_dict[str(img_id)]
        label = metadata.stuff_classes[int(float(label_id))]
        place = [x for x in get_response(agent, f'{caption}, {label}, {(x,y,w,h)}').split(';') if x != '' and x != ' ']
        assert len(place) == 2, f'place = {place}'
        ori_place, gen_place = place[0], place[1]
        
        # id_ = annotation['id']
        img_path =  os.path.join(val_folder, f'{img_id:0{12}}.jpg')
        img_pil = Image.open(img_path)
        image_before_list.append(img_pil)
        opt.edit_txt = f'move {label} from \'{ori_place}\' to \'{gen_place}\''
        
        out_pil = Move_Method(opt, 0, 0, img_pil, preloaded_move_model, preloaded_agent, record_history=False)
        image_after_list.append(out_pil)

        print(f'Images have been moved: {len(selected_list)}')

    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    print(f"clip directional similarity: {clip_directional_similarity}")
    with open("models/clip_directional_similarity_Move.txt", "w") as f:
        f.write(str(clip_directional_similarity))

    fid_score = cal_fid(image_before_list, image_after_list)
    print(f"FID Score: {fid_score}")
    with open("models/fid_score_Move.txt", "w") as f:
        f.write(str(fid_score))

    del preloaded_move_model, preloaded_agents

    # Read MSCOCO


def main():
    
    opt = get_arguments()
    setattr(opt, 'test_group_num', 1)
    
    opt.out_dir = '../autodl-tmp/Exp_Replace'
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    # print('Start to valuate Replace Method...')
    # Val_Replace_Method(opt)
    
    opt.out_dir = '../autodl-tmp/Exp_Move'
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Move Method...')
    Val_Move_Method(opt)


if __name__ == '__main__':
    main()