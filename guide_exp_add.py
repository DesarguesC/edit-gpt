import os, time, json, logging
from prompt.guide import get_response, get_bot
from basicsr.utils import tensor2img, img2tensor
from random import randint, choice
from PIL import Image, ImageOps
import numpy as np
from task_planning import Replace_Method, Move_Method
from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from detectron2.data import MetadataCatalog
from preload_utils import *

from operations.vqa_utils import preload_vqa_model, Val_add_amount

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


def preload_add_model(opt):
    return {
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt) # 10446 MiB
    }

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



def Val_Add_Method(opt):
    val_fodler = '../autodl-tmp/COCO/val2017'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    caption_before_list = caption_after_list = []
    image_before_list = image_after_list = []

    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_val = json.load(f)    
    # query caption via image_id
    selected_list = []
    length = len(data_val['annotations'])

    caption_before_list = caption_after_list = []
    image_before_list = image_after_list = []

    locations = ['left', 'right', 'behind', 'head']
    preloaded_add_model = preload_add_model(opt) if opt.preload_all_models else None
    preloaded_agent = preload_all_agents(opt) if opt.preload_all_models else None
    model_dict = preload_vqa_model(opt.vqa_model_path, opt.device)

    if not os.path.exists(f'{opt.out_dir}/Inputs-Add/'):
        os.mkdir(f'{opt.out_dir}/Inputs-Add/')
    
    while len(selected_list) < opt.test_group_num:

        start_time = time.time()

        idx = randint(0, length)
        while idx in selected_list:
            idx = randint(0, length)

        try:
            selected_list.append(idx)
            instance = data_val['annotations'][idx]
            category_id = int(instance['category_id'])
            img_id = data_val['image_id']
            caption1 = data_val['caption']
            add_label_id = category_id
            while add_label_id == category_id:
                add_label_id = randint(1,81)

            ori_label = metadata.stuff_classes[category_id]
            add_label = metadata.stuff_classes[add_label_id]

            img_path =  os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
            ori_img.save(f'{opt.out_dir}/Inputs-Add/{img_id:0{12}}.jpg')

            amend = f'{choice(location)} of the {ori_label}'
            opt.edit_txt = f'add a {add_label} on tht {amend}'
            captions2 = f'{captions1}, with a {add_label} added on the {amend}'
            out_pil = Add_Method(opt, 0, 0, input_pil, preloaded_add_model, preloaded_agent, record_history=False)

            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            caption_before_list.append(caption1)
            caption_after_list.append(captions2)

            get_amount = Val_add_amount(model_dict, add_label, ori_img, out_pil)
            ac_or_not = 1 if int(float(get_amout)) == 1 else 0
            acc_num = acc_num + ac_or_not

            end_time = time.time()
            string = f'Images have been moved: {len(selected_list)} | Acc: {True if ac_or_not==1 else False} | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)
        
        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    
    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    print(f"clip directional similarity: {clip_directional_similarity}")
    with open("models/clip_directional_similarity_Add.txt", "w") as f:
        f.write(str(clip_directional_similarity))

    fid_score = cal_fid(image_before_list, image_after_list)
    print(f"FID Score: {fid_score}")
    with open("models/fid_score_Add.txt", "w") as f:
        f.write(str(fid_score))

    del preloaded_add_model, preloaded_agent
    
    acc_ratio = acc_ num / len(selected_list)
    print(f"Acc: {acc_ratio}")
     with open("models/acc_ratio_Add.txt", "w") as f:
        f.write(str(acc_ratio))



def main():
    
    opt = get_arguments()
    setattr(opt, 'test_group_num', 2)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Add&Remove.log'
    )
    if os.path.isfile('Add&Remove.log'): os.system('Add&Remove.log')
    
    opt.out_dir = '../autodl-tmp/Exp_Add/'
    if os.path.exists(opt.out_dir): os.system('rm -rf ' + opt.out_dir)
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Add Method...')
    Val_Add_Method(opt)
    


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')