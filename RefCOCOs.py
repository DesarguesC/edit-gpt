import os, time, json, logging
from random import randint, choice
from PIL import Image, ImageOps
from socket import *
import numpy as np
from tqdm import tqdm
import _pickle as pickle
from task_planning import Replace_Method, Move_Method, Transfer_Method
from operations.vqa_utils import A_IsReplacedWith_B, preload_vqa_model
from prompt.guide import get_response, get_bot, system_prompt_gen_move_instructions, system_prompt_edit_sort
from task_planning import Add_Method, Remove_Method, Transfer_Method
from prompt.arguments import get_arguments
from prompt.util import write_instruction, write_valuation_results, cal_metrics_write
from preload_utils import *
from operations.vqa_utils import preload_vqa_model, Val_add_amount, IsRemoved
from pytorch_lightning import seed_everything

def use_exp_agent(opt, system_prompt):
    agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=system_prompt, proxy=opt.net_proxy,
                    type=opt.llm_type)
    return agent

def read_original_prompt(path_to_json):
    assert path_to_json.endswith('.json')
    with open(path_to_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompt1 = data['input']
    edit = data['edit']
    prompt2 = f'{prompt1}, with {edit}'
    return (prompt1, prompt2, edit)


def Val_Replace_Method(opt, preloaded_models=None, preloaded_agents=None, clientSocket=None):
    seed_everything(opt.seed)
    # agent = use_exp_agent(opt, system_prompt_edit_sort)
    val_folder = '../autodl-tmp/COCO/train2017'
    ref_file = pickle.load(open('../autodl-tmp/RefCOCOs/refcoco/refs(unc).p', 'rb'))
    # refcoco/refs(unc).p, refcocog/refs(umd).p, refcoco+/refs(unc).p
    # TODO: create a dic, query certain instance via image-id.
    ref_instance = {}
    all_image_id = []
    for item in ref_file:
        image_id = str(item['image_id'])
        if image_id not in all_image_id:
            all_image_id.append(image_id)
        if image_id not in ref_instance:
            ref_instance[image_id] = item['sentences'][0]['raw'].lower()
        else:
            continue

    with open('../autodl-tmp/COCO/annotations/instances_train2017.json') as f:
        data_ = json.load(f)
        # query caption via image_id

    length = len(all_image_id)
    print(f'all_image_id length = {length}')
    selected_list = []


    with open('../autodl-tmp/COCO/annotations/captions_train2017.json') as f:
        captions = json.load(f)

    captions_dict = {}
    for idx in tqdm(range(len(captions['annotations']))):
        x = captions['annotations'][idx]
        image_id = str(x['image_id'])
        if image_id not in ref_instance:
            continue
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    label_metadata = {}
    for x in data_['categories']:
        label_metadata[str(x['id'])] = x['name']

    # print(f'label_metadata = \n\t{label_metadata}')
    print('\nFile Preloaded...\n')

    image_before_list, image_after_list, image_ip2p_list = [], [], []
    caption_before_list, caption_after_list = [], []
    acc_num_replace, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

    # 4-6 images in a folder
    model_dict = preload_vqa_model(opt.vqa_model_path, opt.device) # prepare VQA validation
    while len(selected_list) < opt.test_group_num:
        start_time = time.time()
        idx = randint(0, length - 1)
        while idx in selected_list:
            idx = randint(0, length - 1)
        selected_list.append(idx)
        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            img_id = all_image_id[idx]
            img_path = os.path.join(val_folder, f'{int(img_id):0{12}}.jpg')
            label_new_id = randint(1, 80)
            label_new = label_metadata[str(label_new_id)]

            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path
            label_ori = ref_instance[str(img_id)]
            opt.edit_txt = f'replace {label_ori} with {label_new}'
            caption1 = captions_dict[str(img_id)]
            caption2 = f'{caption1}; with {label_ori} replaced with {label_new}'

            # print('All Settings are DONE, no model and continue!' + '\n'*2)
            # continue

            out_pil = Replace_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents, record_history=False)
            if out_pil.size != (512, 512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents,
                                           record_history=False, model_type=opt.model_type, clientSocket=clientSocket,
                                           size=(512, 512))
                if out_ip2p.size != (512, 512):
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

            amount_list = A_IsReplacedWith_B(model_dict, label_ori, label_new, ori_img,
                                             [out_pil, out_ip2p] if opt.with_ip2p_val else out_pil, opt.device)
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
    acc_ratio_replace = acc_num_replace / len(selected_list)
    if opt.with_ip2p_val:
        acc_ratio_ip2p = acc_num_ip2p / len(selected_list)
    # consider if there is need to save all images replaced

    string = f'Replace Acc: \n\tEditGPT = {acc_ratio_replace}\n' + (
        f'\t{opt.model_type} = {acc_ratio_ip2p}\n' if opt.with_ip2p_val else '')
    print(string)
    logging.info(string)
    cal_metrics_write(
        image_before_list, image_after_list,
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list,
        caption_after_list, static_out_dir=static_out_dir,
        type_name='Replace', extra_string=string, model_type=opt.model_type
    )

def Val_Remove_Method(opt, preloaded_models=None, preloaded_agents=None, clientSocket=None):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/train2017'
    ref_file = pickle.load(open('../autodl-tmp/RefCOCOs/refcoco/refs(unc).p', 'rb'))
    # TODO: create a dic, query certain instance via image-id.
    ref_instance = {}
    all_image_id = []
    for item in ref_file:
        image_id = str(item['image_id'])
        if image_id not in all_image_id:
            all_image_id.append(image_id)
        if image_id not in ref_instance:
            ref_instance[image_id] = item['sentences'][0]['raw'].lower()
        else:
            continue
    with open('../autodl-tmp/COCO/annotations/instances_train2017.json') as f:
        data_val = json.load(f)
        # query caption via image_id
    selected_list = []
    length = len(all_image_id)
    print(f'all_image_id length = {length}')
    print(f'ref_instances length = {len(ref_instance)}')

    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

    if not os.path.exists(f'{opt.out_dir}/Inputs-Add/'):
        os.mkdir(f'{opt.out_dir}/Inputs-Add/')

    with open('../autodl-tmp/COCO/annotations/captions_train2017.json') as f:
        captions = json.load(f)

    captions_dict = {}
    for idx in tqdm(range(len(captions['annotations']))):
        x = captions['annotations'][idx]
        image_id = str(x['image_id'])
        if image_id not in ref_instance: # i/o speed: in dict >> in list
            continue
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']


    label_metadata = {}
    for x in data_val['categories']:
        label_metadata[str(x['id'])] = x['name']

    acc_num_remove, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

    model_dict = preload_vqa_model(opt.vqa_model_path, opt.device)  # prepare VQA validation
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
            img_id = all_image_id[idx]
            caption1 = captions_dict[str(img_id)]

            ori_label = ref_instance[str(img_id)]
            img_path = os.path.join(val_folder, f'{int(img_id):0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path

            opt.edit_txt = f'remove the {ori_label}'
            caption2 = f'{caption1}; with {ori_label} removed'

            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents,
                                           record_history=False, model_type=opt.model_type, clientSocket=clientSocket, size=(512,512))
                if out_ip2p.size != (512, 512):
                    out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            out_pil = Remove_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents, record_history=False)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            if opt.with_ip2p_val:
                image_ip2p_list.append(out_ip2p)
            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            if opt.with_ip2p_val:
                out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-{opt.model_type}.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)

            amount_list = IsRemoved(model_dict, ori_label, ori_img, [out_pil, out_ip2p] if opt.with_ip2p_val else out_pil, device=opt.device)
            if opt.with_ip2p_val:
                if len(amount_list) != 2:
                    string__ = f"Invalid Val_add_amount in VQA return: len(amount_list) = {len(amount_list)}"
                    print(string__)
                    logging.warning(string__)

                get_amount_remove, get_amount_ip2p = amount_list[0], amount_list[1]
                ac_or_not_remove = 1 if int(float(get_amount_remove)) == 1 else 0
                ac_or_not_ip2p = 1 if int(float(get_amount_ip2p)) == 1 else 0
                acc_num_remove = acc_num_remove + ac_or_not_remove
                acc_num_ip2p = acc_num_ip2p + ac_or_not_ip2p
            else:
                assert not isinstance(amount_list, list)
                ac_or_not_remove = 1 if int(float(amount_list)) == 1 else 0
                acc_num_remove = acc_num_remove + ac_or_not_remove

            end_time = time.time()
            string = (f'Images have been removed: {len(selected_list)} | Acc: [EditGPT/{opt.model_type}]~[{True if ac_or_not_remove == 1 else False} '
                      f'|{True if ac_or_not_ip2p == 1 else False}] | Time cost: {end_time - start_time}') if opt.with_ip2p_val else \
                      f'Images have been removed: {len(selected_list)} | Acc: [EditGPT]~[{True if ac_or_not_remove == 1 else False}] | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM
    # consider if there is need to save all images replaced
    acc_ratio_remove = acc_num_remove / len(selected_list)
    if opt.with_ip2p_val:
        acc_ratio_ip2p = acc_num_ip2p / len(selected_list)

    string = f'Remove Acc: \n\tEditGPT = {acc_ratio_remove}\n' + (f'\t{opt.model_type} = {acc_ratio_ip2p}\n' if opt.with_ip2p_val else '')
    print(string)
    logging.info(string)
    cal_metrics_write(
        image_before_list, image_after_list, 
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list, 
        caption_after_list, static_out_dir=static_out_dir, 
        type_name='Remove', extra_string=string, model_type=opt.model_type
    )




def Val_Replace_Method_g(opt, preloaded_models=None, preloaded_agents=None, clientSocket=None):
    seed_everything(opt.seed)
    # agent = use_exp_agent(opt, system_prompt_edit_sort)
    val_folder = '../autodl-tmp/COCO/train2017'
    ref_file = pickle.load(open('../autodl-tmp/RefCOCOs/refcocog/refs(umd).p', 'rb'))
    # refcoco/refs(unc).p, refcocog/refs(umd).p, refcoco+/refs(unc).p
    # TODO: create a dic, query certain instance via image-id.
    ref_instance = {}
    all_image_id = []
    for item in ref_file:
        image_id = str(item['image_id'])
        if image_id not in all_image_id:
            all_image_id.append(image_id)
        if image_id not in ref_instance:
            ref_instance[image_id] = item['sentences'][0]['raw'].lower()
        else:
            continue

    with open('../autodl-tmp/COCO/annotations/instances_train2017.json') as f:
        data_ = json.load(f)
        # query caption via image_id

    length = len(all_image_id)
    print(f'all_image_id length = {length}')
    selected_list = []


    with open('../autodl-tmp/COCO/annotations/captions_train2017.json') as f:
        captions = json.load(f)

    captions_dict = {}
    for idx in tqdm(range(len(captions['annotations']))):
        x = captions['annotations'][idx]
        image_id = str(x['image_id'])
        if image_id not in ref_instance:
            continue
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    label_metadata = {}
    for x in data_['categories']:
        label_metadata[str(x['id'])] = x['name']

    # print(f'label_metadata = \n\t{label_metadata}')
    print('\nFile Preloaded...\n')

    image_before_list, image_after_list, image_ip2p_list = [], [], []
    caption_before_list, caption_after_list = [], []
    acc_num_replace, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

    # 4-6 images in a folder
    model_dict = preload_vqa_model(opt.vqa_model_path, opt.device) # prepare VQA validation
    while len(selected_list) < opt.test_group_num:
        start_time = time.time()
        idx = randint(0, length - 1)
        while idx in selected_list:
            idx = randint(0, length - 1)
        selected_list.append(idx)
        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            img_id = all_image_id[idx]
            img_path = os.path.join(val_folder, f'{int(img_id):0{12}}.jpg')
            label_new_id = randint(1, 80)
            label_new = label_metadata[str(label_new_id)]

            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path
            label_ori = ref_instance[str(img_id)]
            opt.edit_txt = f'replace {label_ori} with {label_new}'
            caption1 = captions_dict[str(img_id)]
            caption2 = f'{caption1}; with {label_ori} replaced with {label_new}'

            # print('All Settings are DONE, no model and continue!' + '\n'*2)
            # continue

            out_pil = Replace_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents, record_history=False)
            if out_pil.size != (512, 512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents,
                                           record_history=False, model_type=opt.model_type, clientSocket=clientSocket,
                                           size=(512, 512))
                if out_ip2p.size != (512, 512):
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

            amount_list = A_IsReplacedWith_B(model_dict, label_ori, label_new, ori_img,
                                             [out_pil, out_ip2p] if opt.with_ip2p_val else out_pil, opt.device)
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
    acc_ratio_replace = acc_num_replace / len(selected_list)
    if opt.with_ip2p_val:
        acc_ratio_ip2p = acc_num_ip2p / len(selected_list)
    # consider if there is need to save all images replaced

    string = f'Replace Acc: \n\tEditGPT = {acc_ratio_replace}\n' + (
        f'\t{opt.model_type} = {acc_ratio_ip2p}\n' if opt.with_ip2p_val else '')
    print(string)
    logging.info(string)
    cal_metrics_write(
        image_before_list, image_after_list,
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list,
        caption_after_list, static_out_dir=static_out_dir,
        type_name='Replace', extra_string=string, model_type=opt.model_type
    )

def Val_Remove_Method_g(opt, preloaded_models=None, preloaded_agents=None, clientSocket=None):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/train2017'
    ref_file = pickle.load(open('../autodl-tmp/RefCOCOs/refcocog/refs(umd).p', 'rb'))
    # TODO: create a dic, query certain instance via image-id.
    ref_instance = {}
    all_image_id = []
    for item in ref_file:
        image_id = str(item['image_id'])
        if image_id not in all_image_id:
            all_image_id.append(image_id)
        if image_id not in ref_instance:
            ref_instance[image_id] = item['sentences'][0]['raw'].lower()
        else:
            continue
    with open('../autodl-tmp/COCO/annotations/instances_train2017.json') as f:
        data_val = json.load(f)
        # query caption via image_id
    selected_list = []
    length = len(all_image_id)
    print(f'all_image_id length = {length}')
    print(f'ref_instances length = {len(ref_instance)}')

    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

    if not os.path.exists(f'{opt.out_dir}/Inputs-Add/'):
        os.mkdir(f'{opt.out_dir}/Inputs-Add/')

    with open('../autodl-tmp/COCO/annotations/captions_train2017.json') as f:
        captions = json.load(f)

    captions_dict = {}
    for idx in tqdm(range(len(captions['annotations']))):
        x = captions['annotations'][idx]
        image_id = str(x['image_id'])
        if image_id not in ref_instance: # i/o speed: in dict >> in list
            continue
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']


    label_metadata = {}
    for x in data_val['categories']:
        label_metadata[str(x['id'])] = x['name']

    acc_num_remove, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

    model_dict = preload_vqa_model(opt.vqa_model_path, opt.device)  # prepare VQA validation
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
            img_id = all_image_id[idx]
            caption1 = captions_dict[str(img_id)]

            ori_label = ref_instance[str(img_id)]
            img_path = os.path.join(val_folder, f'{int(img_id):0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path

            opt.edit_txt = f'remove the {ori_label}'
            caption2 = f'{caption1}; with {ori_label} removed'

            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents,
                                           record_history=False, model_type=opt.model_type, clientSocket=clientSocket, size=(512,512))
                if out_ip2p.size != (512, 512):
                    out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            out_pil = Remove_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents, record_history=False)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            if opt.with_ip2p_val:
                image_ip2p_list.append(out_ip2p)
            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            if opt.with_ip2p_val:
                out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-{opt.model_type}.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)

            amount_list = IsRemoved(model_dict, ori_label, ori_img, [out_pil, out_ip2p] if opt.with_ip2p_val else out_pil, device=opt.device)
            if opt.with_ip2p_val:
                if len(amount_list) != 2:
                    string__ = f"Invalid Val_add_amount in VQA return: len(amount_list) = {len(amount_list)}"
                    print(string__)
                    logging.warning(string__)

                get_amount_remove, get_amount_ip2p = amount_list[0], amount_list[1]
                ac_or_not_remove = 1 if int(float(get_amount_remove)) == 1 else 0
                ac_or_not_ip2p = 1 if int(float(get_amount_ip2p)) == 1 else 0
                acc_num_remove = acc_num_remove + ac_or_not_remove
                acc_num_ip2p = acc_num_ip2p + ac_or_not_ip2p
            else:
                assert not isinstance(amount_list, list)
                ac_or_not_remove = 1 if int(float(amount_list)) == 1 else 0
                acc_num_remove = acc_num_remove + ac_or_not_remove

            end_time = time.time()
            string = (f'Images have been removed: {len(selected_list)} | Acc: [EditGPT/{opt.model_type}]~[{True if ac_or_not_remove == 1 else False} '
                      f'|{True if ac_or_not_ip2p == 1 else False}] | Time cost: {end_time - start_time}') if opt.with_ip2p_val else \
                      f'Images have been removed: {len(selected_list)} | Acc: [EditGPT]~[{True if ac_or_not_remove == 1 else False}] | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM
    # consider if there is need to save all images replaced
    acc_ratio_remove = acc_num_remove / len(selected_list)
    if opt.with_ip2p_val:
        acc_ratio_ip2p = acc_num_ip2p / len(selected_list)

    string = f'Remove Acc: \n\tEditGPT = {acc_ratio_remove}\n' + (f'\t{opt.model_type} = {acc_ratio_ip2p}\n' if opt.with_ip2p_val else '')
    print(string)
    logging.info(string)
    cal_metrics_write(
        image_before_list, image_after_list,
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list,
        caption_after_list, static_out_dir=static_out_dir,
        type_name='Remove', extra_string=string, model_type=opt.model_type
    )

def main1_g(general_path, opt, preloaded_models=None, preloaded_agents=None, test_group_num=50, clientSocket=None):

    if os.path.isfile('Replace_Move.log'): os.system('Replace_Move.log')
    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename='Replace_Move.log'
    )

    opt.out_dir = os.path.join(general_path, 'Exp_Replace')
    if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Replace Method...')
    Val_Replace_Method(opt, preloaded_models, preloaded_agents, clientSocket)

    # opt.out_dir = os.path.join(general_path, 'Exp_Move')
    # if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    # base_cnt = len(os.listdir(opt.out_dir))
    # setattr(opt, 'base_cnt', base_cnt)
    # print('Start to valuate Move Method...')
    # Val_Move_Method(opt, preloaded_models, preloaded_agents, clientSocket)


def main2_g(general_path, opt, preloaded_models=None, preloaded_agents=None, test_group_num=50, clientSocket=None):
    if os.path.isfile('Add_Remove.log'): os.system('rm Add_Remove.log')

    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename='Add_Remove.log'
    )

    # opt.out_dir = os.path.join(general_path, 'Exp_Add')
    # if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    # base_cnt = len(os.listdir(opt.out_dir))
    # setattr(opt, 'base_cnt', base_cnt)
    # print('Start to valuate Add Method...')
    # Val_Add_Method(opt, preloaded_models, preloaded_agents, clientSocket)

    opt.out_dir = os.path.join(general_path, 'Exp_Remove')
    if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Remove Method...')
    Val_Remove_Method(opt, preloaded_models, preloaded_agents, clientSocket)


def main1(general_path, opt, preloaded_models=None, preloaded_agents=None, test_group_num=50, clientSocket=None):
    if os.path.isfile('Replace_Move.log'): os.system('Replace_Move.log')
    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename='Replace_Move.log'
    )

    opt.out_dir = os.path.join(general_path, 'Exp_Replace')
    if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Replace Method...')
    Val_Replace_Method(opt, preloaded_models, preloaded_agents, clientSocket)

    # opt.out_dir = os.path.join(general_path, 'Exp_Move')
    # if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    # base_cnt = len(os.listdir(opt.out_dir))
    # setattr(opt, 'base_cnt', base_cnt)
    # print('Start to valuate Move Method...')
    # Val_Move_Method(opt, preloaded_models, preloaded_agents, clientSocket)





def main2(general_path, opt, preloaded_models=None, preloaded_agents=None, test_group_num=50, clientSocket=None):

    if os.path.isfile('Add_Remove.log'): os.system('rm Add_Remove.log')

    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Add_Remove.log'
    )
    
    # opt.out_dir = os.path.join(general_path, 'Exp_Add')
    # if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    # base_cnt = len(os.listdir(opt.out_dir))
    # setattr(opt, 'base_cnt', base_cnt)
    # print('Start to valuate Add Method...')
    # Val_Add_Method(opt, preloaded_models, preloaded_agents, clientSocket)

    opt.out_dir = os.path.join(general_path, 'Exp_Remove')
    if not os.path.exists(opt.out_dir): os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Remove Method...')
    Val_Remove_Method(opt, preloaded_models, preloaded_agents, clientSocket)
    

if __name__ == '__main__':
    start_time = time.time()
    opt = get_arguments()
    general_path = opt.out_dir

    if os.path.exists(opt.out_dir):
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)

    clientSocket = None
    if opt.model_type == 'MGIE' and opt.with_ip2p_val:
        clientHost, clientPort = '127.0.0.1', 4096
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((clientHost, clientPort))
    from preload_utils import preload_all_agents, preload_all_models
    preloaded_models = preload_all_models(opt) # if opt.preload_all_models else None
    preloaded_agents = preload_all_agents(opt) # if opt.preload_all_agents else None

    print('\n\nFirst: Replace & Move \n\n')
    main1(general_path, opt, preloaded_models, preloaded_agents, test_group_num=50, clientSocket=clientSocket)
    print('\n\nSecond: Add & Remove \n\n')
    main2(general_path, opt, preloaded_models, preloaded_agents, test_group_num=50, clientSocket=clientSocket)

    opt.out_dir = '../autodl-tmp/exp_RefCOCOg_to_sdedit'
    general_path = opt.out_dir

    if os.path.exists(opt.out_dir):
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    print('\n\nFirst: Replace & Move \n\n')
    main1_g(general_path, opt, preloaded_models, preloaded_agents, test_group_num=50, clientSocket=clientSocket)
    print('\n\nSecond: Add & Remove \n\n')
    main2_g(general_path, opt, preloaded_models, preloaded_agents, test_group_num=50, clientSocket=clientSocket)

    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')