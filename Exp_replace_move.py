import os, time, json, logging
from operations.vqa_utils import A_IsReplacedWith_B, preload_vqa_model
from prompt.guide import get_response, get_bot, system_prompt_gen_move_instructions, system_prompt_edit_sort
from basicsr.utils import tensor2img, img2tensor
from random import randint
from PIL import Image, ImageOps
import numpy as np
from task_planning import Replace_Method, Move_Method, Transfer_Method
from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import calculate_clip_score, PSNR_compute, SSIM_compute, write_instruction, write_valuation_results, cal_metrics_write
from preload_utils import *
from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial
from pytorch_lightning import seed_everything

def preload_replace_model(opt):
    return {
        'preloaded_example_generator': preload_example_generator(opt), 
        # XL - 8272 MiB, XL_ad - 8458 MiB, V1.5 - 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt) if opt.with_ip2p_val else None,  # 8854 MiB
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt), # 10446 MiB
        'preloaded_refiner': preload_refiner(opt) if opt.example_type != 'XL' else None
    }

def preload_move_model(opt):
    return {
        'preloaded_example_generator': preload_example_generator(opt), 
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt) if opt.with_ip2p_val else None,  # 8854 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt), # 10446 MiB
        'preloaded_refiner': preload_refiner(opt) if opt.example_type != 'XL' else None
    }

def use_exp_agent(opt, system_prompt):
    agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=system_prompt, proxy=opt.net_proxy, type=opt.llm_type)
    return agent


def read_original_prompt(path_to_json):
    assert path_to_json.endswith('.json')
    with open(path_to_json, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    prompt1 = data['input']
    edit = data['edit']
    prompt2 = f'{prompt1}, with {edit}'
    return (prompt1, prompt2, edit)

def Val_Replace_Method(opt, clientSocket):
    seed_everything(opt.seed)
    # agent = use_exp_agent(opt, system_prompt_edit_sort)
    val_folder = '../autodl-tmp/COCO/val2017'
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_ = json.load(f)
        # query caption via image_id
    data_val = data_['annotations']
    length = len(data_val)
    folders = os.listdir(val_folder)
    length = len(folders)
    selected_list = []

    from preload_utils import preload_all_agents
    preloaded_replace_model = preload_replace_model(opt) if opt.preload_all_models else None
    preloaded_agent = preload_all_agents(opt) if opt.preload_all_agents else None
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
    
    label_metadata = {}
    for x in data_['categories']:
        label_metadata[str(x['id'])] = x['name']

    image_before_list,  image_after_list,  image_ip2p_list = [], [], []
    caption_before_list, caption_after_list = [], []
    acc_num_replace, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

    # 4-6 images in a folder
    while len(selected_list) < opt.test_group_num:
        start_time = time.time()
        idx = randint(0,length-1)
        while idx in selected_list:
            idx = randint(0, length-1)
        selected_list.append(idx)
        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            instance = data_val[idx]
            label_id = int(instance['category_id'])
            label_new_id = randint(1, 80)
            while (label_new_id == label_id) or (str(label_new_id) not in label_metadata.keys()):
                label_new_id = randint(1, 80)
            label_ori = label_metadata[str(label_id)]
            label_new = label_metadata[str(label_new_id)]
            print(f'(label_id, label_ori) = {(label_id, label_ori)}, (label_new_id, label_new) = {(label_new_id, label_new)}')

            img_id = instance['image_id']
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')

            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.edit_txt = f'replace {label_ori} with {label_new}'
            caption1 = captions_dict[str(img_id)]
            caption2 = f'{caption1}; with {label_ori} replaced with {label_new}'

            out_pil = Replace_Method(opt, 0, 0, ori_img, preloaded_replace_model, preloaded_agent, record_history=False)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_replace_model, preloaded_agent, record_history=False, model_type=opt.model_type, clientSocket=clientSocket, size=(512,512))
                if out_ip2p.size != (512,512):
                    out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            if opt.with_ip2p_val:
                out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-{opt.model_type}.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)

            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            if opt.with_ip2p_val:
                image_ip2p_list.append(out_ip2p)

            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            amount_list = A_IsReplacedWith_B(model_dict, label_ori, label_new, ori_img, [out_pil, out_ip2p] if opt.with_ip2p_val else out_pil, opt.device)
            if opt.with_ip2p_val:
                if len(amount_list) != 2:
                    string__ = f"Invalid Val_add_amount in VQA return: len(amount_list) = {len(amount_list)}"
                    print(string__)
                    logging.warning(string__)
                a, b = amount_list[0], amount_list[1]
                acc_num_replace += a
                acc_num_ip2p += b
            else:
                assert not isinstance(amount_list, list)
                acc_num_replace += amount_list

            end_time = time.time()

            string = (
                f'Images have been replaced: {len(selected_list)} | Acc: [EditGPT/{opt.model_type}]~[{True if a == 1 else False}|'
                f'{True if b == 1 else False}] | Time cost: {end_time - start_time}') if opt.with_ip2p_val else \
                f'Images have been replaced: {len(selected_list)} | Acc: [EditGPT] ~ [{True if amount_list == 1 else False}] | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM

    del preloaded_agent, preloaded_replace_model
    acc_ratio_replace = acc_num_replace / len(selected_list)
    if opt.with_ip2p_val:
        acc_ratio_ip2p = acc_num_ip2p / len(selected_list)
    # consider if there is need to save all images replaced
    
    string = f'Replace Acc: \n\tEditGPT = {acc_ratio_replace}\n' + (f'\t{opt.model_type} = {acc_ratio_ip2p}\n' if opt.with_ip2p_val else '')
    print(string)
    logging.info(string)
    cal_metrics_write(
        image_before_list, image_after_list, 
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list, 
        caption_after_list, static_out_dir=static_out_dir, 
        type_name='Replace', extra_string=string, model_type=opt.model_type
    )

def Val_Move_Method(opt, clientSocket):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/val2017/'
    # Mute GPT
    agent = use_exp_agent(opt, system_prompt_gen_move_instructions)


    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        caption = json.load(f)    
    # query caption via image_id
    captions_dict = {}
    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

    preloaded_move_model = preload_move_model(opt) if opt.preload_all_models else None
    preloaded_agent = preload_all_agents(opt) if opt.preload_all_agents else None

    for x in caption['annotations']:
        image_id = str(x['image_id'])
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data = json.load(f)
    label_metadata = {}
    for x in data['categories']:
        label_metadata[str(x['id'])] = x['name']
    
    length = len(data['annotations'])
    selected_list = []
    
    static_out_dir = opt.out_dir
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)

    while len(selected_list) < opt.test_group_num:
        start_time = time.time()

        idx = randint(0, length-1)
        while idx in selected_list:
            idx = randint(0, length-1)
        selected_list.append(idx)

        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            annotation = data['annotations'][idx]
            start_time = time.time()

            # be used in those Mute GOT modules
            x, y, w, h = annotation['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            img_id, label_id = annotation['image_id'], annotation['category_id']
            caption = captions_dict[str(img_id)]
            label = label_metadata[str(label_id)]

            # Mute GPT
            place = [x for x in get_response(agent, f'{opt.edit_txt}, {label}, {(x,y,w,h)}').split(';') if x != '' and x != ' ']
            # place = ['on the right', 'on the left']

            assert len(place) == 2, f'place = {place}'
            ori_place, gen_place = place[0], place[1]

            opt.edit_txt = f'move {label} from \'{ori_place}\' to \'{gen_place}\'' # regularized edit_txt
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            img_pil = ImageOps.fit(Image.open(img_path).convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)

            out_pil = Move_Method(opt, 0, 0, img_pil, preloaded_move_model, preloaded_agent, record_history=False, target=label)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil, (512,512), method=Image.Resampling.LANCZOS)
            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, img_pil, preloaded_move_model, preloaded_agent, record_history=False, model_type=opt.model_type, clientSocket=clientSocket, size=(512,512))
                if out_ip2p.size != (512,512):
                    out_ip2p = ImageOps.fit(out_ip2p, (512,512), method=Image.Resampling.LANCZOS)

            end_time = time.time()
            string = f'Images have been moved: {len(selected_list)} | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

            image_before_list.append(img_pil)
            image_after_list.append(out_pil)
            if opt.with_ip2p_val:
                image_ip2p_list.append(out_ip2p)
            c1, c2 = f'{caption}; {label} at \'{ori_place}\'', f'{caption}; {label} at \'{gen_place}\''
            caption_before_list.append(c1)
            caption_after_list.append(c2)

            img_pil.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            if opt.with_ip2p_val:
                out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-{opt.model_type}.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', c1, c2, opt.edit_txt)

            end_time = time.time()
            string = (f'Images have been moved: {len(selected_list)} | Time cose: {end_time - start_time}')
            print(string)
            logging.info(string)

        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]
    
    del preloaded_agent, preloaded_move_model
    # consider if there is need to save all images replaced

    # TODO: Clip Image Score & PSNR && SSIM
    cal_metrics_write(
        image_before_list, image_after_list, 
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list, 
        caption_after_list, static_out_dir=static_out_dir, 
        type_name='Move', extra_string=None, model_type=opt.model_type
    )
    
def main1(opt, test_group_num=50, clientSocket=None):

    if os.path.isfile('Replace_Move.log'): os.system('Replace_Move.log')
    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Replace_Move.log'
    )
    
    opt.out_dir = '../autodl-tmp/Exp_Replace'
    if os.path.exists(opt.out_dir):
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Replace Method...')
    Val_Replace_Method(opt, clientSocket)
    
    opt.out_dir = '../autodl-tmp/Exp_Move'
    if os.path.exists(opt.out_dir): 
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Move Method...')
    Val_Move_Method(opt, clientSocket)


if __name__ == '__main__':
    start_time = time.time()
    opt = get_arguments()
    main1(opt, 50)
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')