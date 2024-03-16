import os, time, json, logging
from prompt.guide import get_response, get_bot
from basicsr.utils import tensor2img, img2tensor
from random import randint, choice
from PIL import Image, ImageOps
import numpy as np
from task_planning import Add_Method, Remove_Method, Transfer_Method
from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from detectron2.data import MetadataCatalog
from preload_utils import *

from operations.vqa_utils import preload_vqa_model, Val_add_amount


def preload_add_model(opt):
    return {
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_example_generator': preload_example_generator(opt), 
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt), # 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt), # 8854 MiB
    }

def use_exp_agent(opt, system_prompt):
    agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=system_prompt, proxy=opt.net_proxy)
    return agent

def Val_Add_Method(opt):
    val_folder = '../autodl-tmp/COCO/val2017'
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

    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        captions = json.load(f)    
    
    captions_dict = {}
    for x in captions['annotations']:
        image_id = str(x['image_id'])
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + ', ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    acc_num_add = acc_num_ip2p = 0
    while len(selected_list) < opt.test_group_num:

        start_time = time.time()

        idx = randint(0, length)
        while idx in selected_list:
            idx = randint(0, length)

        try:
            selected_list.append(idx)
            instance = data_val['annotations'][idx]
            category_id = int(instance['category_id'])
            img_id = instance['image_id']
            caption1 = captions_dict[str(img_id)]
            add_label_id = category_id
            while add_label_id == category_id:
                add_label_id = randint(1,81)

            ori_label = metadata.stuff_classes[category_id]
            add_label = metadata.stuff_classes[add_label_id]

            img_path =  os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
            ori_img.save(f'{opt.out_dir}/Inputs-Add/{img_id:0{12}}.jpg')

            amend = f'{choice(locations)} of the {ori_label}'
            opt.edit_txt = f'add a {add_label} on tht {amend}'
            caption2 = f'{caption1}, with a {add_label} added on the {amend}'
            out_pil = Add_Method(opt, 0, 0, ori_img, preloaded_add_model, preloaded_agent, record_history=False)

            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            out_ip2p_pil = Transfer_Method(opt, 0, 0, ori_img, preloaded_add_model, preloaded_agent, record_history=False)

            get_amount_add, get_amount_ip2p = Val_add_amount(model_dict, add_label, ori_img, [out_pil, out_ip2p_pil], device=opt.device)
            ac_or_not_add = 1 if int(float(get_amount_add)) == 1 else 0
            ac_or_not_ip2p = 1 if int(float(get_amount_ip2p)) == 1 else 0
            acc_num_add = acc_num_add + ac_or_not_add
            acc_num_ip2p = acc_num_ip2p + ac_or_not_ip2p

            end_time = time.time()
            string = f'Images have been moved: {len(selected_list)} | Acc: [Add/Ip2p]~[{True if ac_or_not_add==1 else False}|{True if ac_or_not_ip2p==1 else False}] | Time cost: {end_time - start_time}'
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
    
    acc_ratio_add = acc_num_add / len(selected_list)
    acc_ratio_ip2p = acc_num_ip2p / len(selected_list)
    string = f"Acc: [Add|Ip2p]~[{acc_ratio_add}|{acc_ratio_ip2p}]"
    print(string)
    with open("models/acc_ratio_Add_Ip2p.txt", "w") as f:
        f.write(string)



def main():
    
    opt = get_arguments()
    setattr(opt, 'test_group_num', 20)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='\"Add&Remove.log\"'
    )
    if os.path.isfile('\"Add&Remove.log\"'): os.system('\"rm Add&Remove.log\"')
    
    opt.out_dir = '../autodl-tmp/Exp_Add/'
    if os.path.exists(opt.out_dir): os.system('rm -rf ' + opt.out_dir)
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
        os.mkdir(f'{opt.out_dir}/Transfer')
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Add Method...')
    Val_Add_Method(opt)
    


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')