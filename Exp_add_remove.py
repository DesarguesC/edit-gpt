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
from prompt.util import calculate_clip_score, PSNR_compute, SSIM_compute
from detectron2.data import MetadataCatalog
from preload_utils import *

from operations.vqa_utils import preload_vqa_model, Val_add_amount, IsRemoved

from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial
from pytorch_lightning import seed_everything

def preload_add_model(opt):
    return {
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_example_generator': preload_example_generator(opt), 
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt), # 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt), # 8854 MiB
    }

def preload_remove_model(opt):
    return {
        'preloaded_sam_generator': preload_sam_generator(opt),  # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt),  # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt),  # 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt),
    }

def use_exp_agent(opt, system_prompt):
    agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=system_prompt, proxy=opt.net_proxy)
    return agent

def write_instruction(path, caption_before, caption_after, caption_edit):
    """
        line 1: caption_before
        line 2: caption_after
        line 3: caption_edit
    """
    # if not os.path.exists(path): os.mkdir(path)
    # if not path.endswith('txt'):
    #     path = os.path.join(path, 'captions.txt')
    with open(path, 'w') as f:
        f.write(f'{caption_before}\n{caption_after}\n{caption_edit}')

def write_valuation_results(path, typer='', clip_score=None, clip_directional_similarity=None, psnr_score=None, ssim_score=None, fid_score=None, extra_string=None):
    string = (f'Exp For: {typer}\nClip Score: {clip_score}\nClip Directional Similarity: {clip_directional_similarity}\n'
              f'PSNR: {psnr_score}\nSSIM: {ssim_score}\nFID: {fid_score}') + f"\n{extra_string}" if extra_string is not None else ""
    with open(path, 'w') as f:
        f.write(string)
    print(string)


def Val_Add_Method(opt):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/val2017'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_val = json.load(f)    
    # query caption via image_id
    selected_list = []
    length = len(data_val['annotations'])

    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

    # locations = ['left', 'right', 'behind', 'head']
    preloaded_add_model = preload_add_model(opt) if opt.preload_all_models else None
    preloaded_agent = preload_all_agents(opt) if opt.preload_all_models else None
    model_dict = preload_vqa_model(opt.vqa_model_path, opt.device)

    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        captions = json.load(f)    
    
    captions_dict = {}
    for x in captions['annotations']:
        image_id = str(x['image_id'])
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    acc_num_add = acc_num_ip2p = 0
    static_out_dir = opt.out_dir
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)

    while len(selected_list) < opt.test_group_num:

        start_time = time.time()

        idx = randint(0, length)
        while idx in selected_list:
            idx = randint(0, length)
        selected_list.append(idx)

        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            instance = data_val['annotations'][idx]
            category_id = int(instance['category_id'])
            img_id = instance['image_id']
            caption1 = captions_dict[str(img_id)]
            add_label_id = category_id
            while add_label_id == category_id:
                add_label_id = randint(1,81)

            add_label = metadata.stuff_classes[add_label_id]
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)

            # amend = f'{choice(locations)} of the {ori_label}'
            # opt.edit_txt = f'add a {add_label} on tht {amend}'
            # caption2 = f'{caption1}, with a {add_label} added on the {amend}'
            opt.edit_txt = f'add a {add_label}'
            caption2 = f'{caption1}; with {add_label} added'
            out_pil = Add_Method(opt, 0, 0, ori_img, preloaded_add_model, preloaded_agent, record_history=False)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_add_model, preloaded_agent, record_history=False)
            if out_ip2p.size != (512,512):
                out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            image_ip2p_list.append(out_ip2p)
            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            string_ = f"{len(image_before_list), len(image_after_list), len(image_ip2p_list), len(caption_before_list), len(caption_after_list)}"
            print(string_)
            logging.info(string_)

            ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-Ip2p.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)


            amount_list = Val_add_amount(model_dict, add_label, ori_img, [out_pil, out_ip2p], device=opt.device)

            if len(amount_list) != 2:
                string__ = f"Invalid Val_add_amount in VQA return: len(amount_list) = {len(amount_list)}"
                print(string__)
                logging.warning(string__)

            get_amount_add, get_amount_ip2p = amount_list[0], amount_list[1]
            ac_or_not_add = 1 if int(float(get_amount_add)) == 1 else 0
            ac_or_not_ip2p = 1 if int(float(get_amount_ip2p)) == 1 else 0
            acc_num_add = acc_num_add + ac_or_not_add
            acc_num_ip2p = acc_num_ip2p + ac_or_not_ip2p

            end_time = time.time()
            string = (f'Images have been added: {len(selected_list)} | Acc: [EditGPT/Ip2p]~[{True if ac_or_not_add==1 else False}|'
                      f'{True if ac_or_not_ip2p==1 else False}] | Time cost: {end_time - start_time}')
            print(string)
            logging.info(string)
        
        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM

    # use list[Image]
    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    fid_score = cal_fid(image_before_list, image_after_list)

    clip_directional_similarity_ip2p = cal_similarity(image_before_list, image_ip2p_list, caption_before_list, caption_after_list)
    fid_score_ip2p = cal_fid(image_before_list, image_ip2p_list)

    # use list[np.array]
    for i in range(len(image_after_list)):
        image_after_list[i] = np.array(image_after_list[i])
        image_before_list[i] = np.array(image_before_list[i])
        image_ip2p_list[i] = np.array(image_ip2p_list[i])

    ssim_score = SSIM_compute(image_before_list, image_after_list)
    psnr_score = PSNR_compute(image_before_list, image_after_list)

    ssim_score_ip2p = SSIM_compute(image_before_list, image_ip2p_list)
    psnr_score_ip2p = PSNR_compute(image_before_list, image_ip2p_list)

    del preloaded_agent, preloaded_add_model
    # consider if there is need to save all images replaced
    acc_ratio_add, acc_ratio_ip2p = acc_num_add / len(selected_list), acc_num_ip2p / len(selected_list)
    
    clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-large-patch14')
    try:
        clip_score = calculate_clip_score(image_after_list, caption_after_list, clip_score_fn=clip_score_fn)
        clip_score_ip2p = calculate_clip_score(image_ip2p_list, caption_after_list, clip_score_fn=clip_score_fn)
    except Exception as e:
        string = f'Exception Occurred when calculating Clip Score: {e}'
        print(string)
        logging.info(string)
        clip_score = string
        clip_score_ip2p = string
    
    
    string = f'Add Acc: \n\tEditGPT = {acc_ratio_add}\n\tIP2P = {acc_ratio_ip2p}\n'
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Add.txt'), 'Add-EditGPT', clip_score, clip_directional_similarity, psnr_score, ssim_score, fid_score, extra_string=string)
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Ip2p.txt'), 'Add-Ip2p', clip_score_ip2p, clip_directional_similarity_ip2p, psnr_score_ip2p, ssim_score_ip2p, fid_score_ip2p, extra_string=string)

def Val_Remove_Method(opt):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/val2017'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_val = json.load(f)
        # query caption via image_id
    selected_list = []
    length = len(data_val['annotations'])

    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

    preloaded_remove_model = preload_remove_model(opt) if opt.preload_all_models else None
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
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    acc_num_remove = acc_num_ip2p = 0
    static_out_dir = opt.out_dir

    while len(selected_list) < opt.test_group_num:

        start_time = time.time()

        idx = randint(0, length)
        while idx in selected_list:
            idx = randint(0, length)
        selected_list.append(idx)

        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            instance = data_val['annotations'][idx]
            category_id = int(instance['category_id'])
            img_id = instance['image_id']
            caption1 = captions_dict[str(img_id)]
            add_label_id = category_id
            while add_label_id == category_id:
                add_label_id = randint(1, 81)

            ori_label = metadata.stuff_classes[category_id]
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            # amend = f'{choice(locations)} of the {ori_label}'
            # opt.edit_txt = f'add a {add_label} on tht {amend}'
            # caption2 = f'{caption1}, with a {add_label} added on the {amend}'
            opt.edit_txt = f'remove the {ori_label}'
            caption2 = f'{caption1}; with {ori_label} removed'
            out_pil = Remove_Method(opt, 0, 0, ori_img, preloaded_remove_model, preloaded_agent, record_history=False)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_remove_model, preloaded_agent, record_history=False)
            if out_ip2p.size != (512, 512):
                out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)


            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            image_ip2p_list.append(out_ip2p)
            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditPGT.jpg')
            out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-Ip2p.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)

            amount_list = IsRemoved(model_dict, ori_label, ori_img, [out_pil, out_ip2p], device=opt.device)

            if len(amount_list) != 2:
                string__ = f"Invalid Val_add_amount in VQA return: len(amount_list) = {len(amount_list)}"
                print(string__)
                logging.warning(string__)

            get_amount_remove, get_amount_ip2p = amount_list[0], amount_list[1]
            ac_or_not_remove = 1 if int(float(get_amount_remove)) == 1 else 0
            ac_or_not_ip2p = 1 if int(float(get_amount_ip2p)) == 1 else 0
            acc_num_remove = acc_num_remove + ac_or_not_remove
            acc_num_ip2p = acc_num_ip2p + ac_or_not_ip2p

            end_time = time.time()
            string = f'Images have been removed: {len(selected_list)} | Acc: [EditGPT/Ip2p]~[{True if ac_or_not_remove == 1 else False}|{True if ac_or_not_ip2p == 1 else False}] | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM

    # use list[Image]
    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    fid_score = cal_fid(image_before_list, image_after_list)

    clip_directional_similarity_ip2p = cal_similarity(image_before_list, image_ip2p_list, caption_before_list,
                                                 caption_after_list)
    fid_score_ip2p = cal_fid(image_before_list, image_ip2p_list)

    # use list[np.array]
    for i in range(len(image_after_list)):
        image_after_list[i] = np.array(image_after_list[i])
        image_before_list[i] = np.array(image_before_list[i])
        image_ip2p_list[i] = np.array(image_ip2p_list[i])

    ssim_score = SSIM_compute(image_before_list, image_after_list)
    psnr_score = PSNR_compute(image_before_list, image_after_list)

    ssim_score_ip2p = SSIM_compute(image_before_list, image_ip2p_list)
    psnr_score_ip2p = PSNR_compute(image_before_list, image_ip2p_list)

    del preloaded_agent, preloaded_remove_model
    # consider if there is need to save all images replaced
    acc_ratio_remove, acc_ratio_ip2p = acc_num_remove / len(selected_list), acc_num_ip2p / len(selected_list)
    
    clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-large-patch14')
    try:
        clip_score = calculate_clip_score(image_after_list, caption_after_list, clip_score_fn=clip_score_fn)
        clip_score_ip2p = calculate_clip_score(image_ip2p_list, caption_after_list, clip_score_fn=clip_score_fn)
    except Exception as e:
        string = f'Exception Occurred when calculating Clip Score: {e}'
        print(string)
        logging.info(string)
        clip_score = string
        clip_score_ip2p = string
    
    string = f'Remove Acc: \n\tEditGPT = {acc_ratio_remove}\n\tIP2P = {acc_ratio_ip2p}\n'
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Remove.txt'), 'Remove-EditGPT', clip_score,
                            clip_directional_similarity, psnr_score, ssim_score, fid_score, extra_string=string)
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Remove.txt'), 'Remove-Ip2p', clip_score_ip2p,
                            clip_directional_similarity_ip2p, psnr_score_ip2p, ssim_score_ip2p, fid_score_ip2p, extra_string=string)

def main2():

    if os.path.isfile('Add_Remove.log'): os.system('rm Add_Remove.log')
    opt = get_arguments()
    setattr(opt, 'test_group_num', 50)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Add_Remove.log'
    )
    
    opt.out_dir = '../autodl-tmp/Exp_Add'
    if os.path.exists(opt.out_dir): os.system(f'rm {opt.out_dir}.zip && zip -r {opt.out_dir}.zip {opt.out_dir} && rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Add Method...')
    Val_Add_Method(opt)

    opt.out_dir = '../autodl-tmp/Exp_Remove'
    if os.path.exists(opt.out_dir): os.system(f'rm {opt.out_dir}.zip && zip {opt.out_dir}.zip {opt.out_dir} && rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Remove Method...')
    Val_Remove_Method(opt)
    


if __name__ == '__main__':
    start_time = time.time()
    from Exp_replace_move import main1
    print('\nnFirst: Replace & Move \n\n')
    main1()
    print('\n\nSecond: Add & Remove \n\n')
    main2()
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')