import os, time, json, logging
from random import randint, choice
from PIL import Image, ImageOps
from socket import *
import numpy as np
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


def Val_Replace_Method(opt, preloaded_models, preloaded_agents, clientSocket=None):
    seed_everything(opt.seed)
    # agent = use_exp_agent(opt, system_prompt_edit_sort)
    val_folder = '../autodl-tmp/COCO/val2017'
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_ = json.load(f)
        # query caption via image_id
    data_val = data_['annotations']
    folders = os.listdir(val_folder)
    length = len(folders)
    selected_list = []

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

    image_before_list, image_after_list, image_ip2p_list = [], [], []
    caption_before_list, caption_after_list = [], []
    acc_num_replace, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

    # 4-6 images in a folder
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
            instance = data_val[idx]
            label_id = int(instance['category_id'])
            label_new_id = randint(1, 80)
            while (label_new_id == label_id) or (str(label_new_id) not in label_metadata.keys()):
                label_new_id = randint(1, 80)
            label_ori = label_metadata[str(label_id)]
            label_new = label_metadata[str(label_new_id)]
            print(
                f'(label_id, label_ori) = {(label_id, label_ori)}, (label_new_id, label_new) = {(label_new_id, label_new)}')

            img_id = instance['image_id']
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')

            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path

            opt.edit_txt = f'replace {label_ori} with {label_new}'
            caption1 = captions_dict[str(img_id)]
            caption2 = f'{caption1}; with {label_ori} replaced with {label_new}'

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

def Val_Move_Method(opt, preloaded_models, preloaded_agents, clientSocket):
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

        idx = randint(0, length - 1)
        while idx in selected_list:
            idx = randint(0, length - 1)
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
            place = [x for x in get_response(agent, f'{opt.edit_txt}, {label}, {(x, y, w, h)}').split(';') if
                     x != '' and x != ' ']
            # place = ['on the right', 'on the left']

            assert len(place) == 2, f'place = {place}'
            ori_place, gen_place = place[0], place[1]

            opt.edit_txt = f'move {label} from \'{ori_place}\' to \'{gen_place}\''  # regularized edit_txt
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            img_pil = ImageOps.fit(Image.open(img_path).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path

            out_pil = Move_Method(opt, 0, 0, img_pil, preloaded_models, preloaded_agents, record_history=False,
                                  target=label)
            if out_pil.size != (512, 512):
                out_pil = ImageOps.fit(out_pil, (512, 512), method=Image.Resampling.LANCZOS)
            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, img_pil, preloaded_models, preloaded_agents, record_history=False,
                                           model_type=opt.model_type, clientSocket=clientSocket, size=(512, 512))
                if out_ip2p.size != (512, 512):
                    out_ip2p = ImageOps.fit(out_ip2p, (512, 512), method=Image.Resampling.LANCZOS)

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

    # TODO: Clip Image Score & PSNR && SSIM
    cal_metrics_write(
        image_before_list, image_after_list,
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list,
        caption_after_list, static_out_dir=static_out_dir,
        type_name='Move', extra_string=None, model_type=opt.model_type
    )

def Val_Add_Method(opt, preloaded_models, preloaded_agents, clientSocket):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/val2017'
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_val = json.load(f)    
    # query caption via image_id
    selected_list = []
    length = len(data_val['annotations'])

    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

    # locations = ['left', 'right', 'behind', 'head']

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
    for x in data_val['categories']:
        label_metadata[str(x['id'])] = x['name']
            
    acc_num_add, acc_num_ip2p = 0, 0
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
            instance = data_val['annotations'][idx]
            category_id = int(instance['category_id'])
            img_id = instance['image_id']
            caption1 = captions_dict[str(img_id)]
            add_label_id = category_id
            while (add_label_id == category_id) or (str(add_label_id) not in label_metadata.keys()):
                add_label_id = randint(1,80)

            add_label = label_metadata[str(add_label_id)]
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
            opt.in_dir = img_path

            # amend = f'{choice(locations)} of the {ori_label}'
            # opt.edit_txt = f'add a {add_label} on tht {amend}'
            # caption2 = f'{caption1}, with a {add_label} added on the {amend}'
            opt.edit_txt = f'add a {add_label}'
            caption2 = f'{caption1}; with {add_label} added'

            if opt.with_ip2p_val:
                out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents,
                                           record_history=False, model_type=opt.model_type, clientSocket=clientSocket, size=(512,512))
                if out_ip2p.size != (512,512):
                    out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            out_pil = Add_Method(opt, 0, 0, ori_img, preloaded_models, preloaded_agents, record_history=False)
            if out_pil.size != (512,512):
                out_pil = ImageOps.fit(out_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)


            image_before_list.append(ori_img)
            image_after_list.append(out_pil)
            if opt.with_ip2p_val:
                image_ip2p_list.append(out_ip2p)
            caption_before_list.append(caption1)
            caption_after_list.append(caption2)

            # string_ = f"{len(image_before_list), len(image_after_list), len(image_ip2p_list), len(caption_before_list), len(caption_after_list)}"
            # print(string_)
            # logging.info(string_)

            ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            if opt.with_ip2p_val:
                out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-{opt.model_type}.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)


            amount_list = Val_add_amount(model_dict, add_label, ori_img, [out_pil, out_ip2p] if opt.with_ip2p_val else None, device=opt.device)

            if opt.with_ip2p_val:
                if len(amount_list) != 2:
                    string__ = f"Invalid Val_add_amount in VQA return: len(amount_list) = {len(amount_list)}"
                    print(string__)
                    logging.warning(string__)

                get_amount_add, get_amount_ip2p = amount_list[0], amount_list[1]
                ac_or_not_add = 1 if int(float(get_amount_add)) == 1 else 0
                ac_or_not_ip2p = 1 if int(float(get_amount_ip2p)) == 1 else 0
                acc_num_add = acc_num_add + ac_or_not_add
                acc_num_ip2p = acc_num_ip2p + ac_or_not_ip2p
            else:
                assert not isinstance(amount_list, list)
                ac_or_not_add = 1 if int(float(amount_list)) == 1 else 0
                acc_num_add = acc_num_add + ac_or_not_add

            end_time = time.time()
            string = (f'Images have been added: {len(selected_list)} | Acc: [EditGPT/{opt.model_type}]~[{True if ac_or_not_add==1 else False}|'
                      f'{True if ac_or_not_ip2p==1 else False}] | Time cost: {end_time - start_time}') if opt.with_ip2p_val else \
                      f'Images have been added: {len(selected_list)} | Acc: [EditGPT]~[{True if ac_or_not_add==1 else False}] | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)
        
        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM

    # consider if there is need to save all images replaced
    acc_ratio_add = acc_num_add / len(selected_list)
    if opt.with_ip2p_val:
        acc_ratio_ip2p = acc_num_ip2p / len(selected_list)

    string = f'Add Acc: \n\tEditGPT = {acc_ratio_add}\n' + (f'\t{opt.model_type} = {acc_ratio_ip2p}\n' if opt.with_ip2p_val else '')
    print(string)
    logging.info(string)
    cal_metrics_write(
        image_before_list, image_after_list, 
        image_ip2p_list if opt.with_ip2p_val else None, caption_before_list, caption_after_list, static_out_dir=static_out_dir, 
        type_name='Add', extra_string=string, model_type=opt.model_type
    )

def Val_Remove_Method(opt, preloaded_models, preloaded_agents, clientSocket=None):
    seed_everything(opt.seed)
    val_folder = '../autodl-tmp/COCO/val2017'
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_val = json.load(f)
        # query caption via image_id
    selected_list = []
    length = len(data_val['annotations'])

    caption_before_list, caption_after_list = [], []
    image_before_list, image_after_list, image_ip2p_list = [], [], []

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
    label_metadata = {}
    for x in data_val['categories']:
        label_metadata[str(x['id'])] = x['name']

    acc_num_remove, acc_num_ip2p = 0, 0
    static_out_dir = opt.out_dir

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
            instance = data_val['annotations'][idx]
            category_id = int(instance['category_id'])
            img_id = instance['image_id']
            caption1 = captions_dict[str(img_id)]
            add_label_id = category_id
            while (add_label_id == category_id) or (str(add_label_id) not in label_metadata.keys()):
                add_label_id = randint(1, 80)

            ori_label = label_metadata[str(category_id)]
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
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


def main1(general_path, opt, preloaded_models, preloaded_agents, test_group_num=50, clientSocket=None):
    if os.path.isfile('Replace_Move.log'): os.system('Replace_Move.log')
    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename='Replace_Move.log'
    )

    opt.out_dir = os.path.join(general_path, 'Exp_Replace')
    print(f'in main1: opt.out_dir = {opt.out_dir}')
    if os.path.exists(opt.out_dir):
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Replace Method...')
    Val_Replace_Method(opt, preloaded_models, preloaded_agents, clientSocket)

    opt.out_dir = os.path.join(general_path, 'Exp_Move')
    if os.path.exists(opt.out_dir):
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Move Method...')
    Val_Move_Method(opt, preloaded_models, preloaded_agents, clientSocket)

def main2(general_path, opt, preloaded_models, preloaded_agents, test_group_num=50, clientSocket=None):

    if os.path.isfile('Add_Remove.log'): os.system('rm Add_Remove.log')

    setattr(opt, 'test_group_num', test_group_num)
    seed_everything(opt.seed)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Add_Remove.log'
    )
    
    opt.out_dir = os.path.join(general_path, 'Exp_Add')
    if os.path.exists(opt.out_dir): 
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Add Method...')
    Val_Add_Method(opt, preloaded_models, preloaded_agents, clientSocket)

    opt.out_dir = os.path.join(general_path, 'Exp_Remove')
    if os.path.exists(opt.out_dir): 
        os.system(f'rm {opt.out_dir}.zip')
        os.system(f'zip -r {opt.out_dir}.zip {opt.out_dir}')
        os.system(f'rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Remove Method...')
    Val_Remove_Method(opt, preloaded_models, preloaded_agents, clientSocket)
    

if __name__ == '__main__':
    start_time = time.time()
    opt = get_arguments()
    general_path = opt.out_dir

    clientSocket = None
    if opt.model_type == 'MGIE':
        clientHost, clientPort = '127.0.0.1', 4096
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((clientHost, clientPort))
    from preload_utils import preload_all_agents, preload_all_models
    preloaded_models = preload_all_models(opt)
    preloaded_agents = preload_all_agents(opt)

    print('\n\nFirst: Replace & Move \n\n')
    main1(general_path, opt, preloaded_models, preloaded_agents, test_group_num=1, clientSocket=clientSocket)
    print('\n\nSecond: Add & Remove \n\n')
    main2(general_path, opt, preloaded_models, preloaded_agents, test_group_num=1, clientSocket=clientSocket)
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')