import os, time, json, logging
from operations.vqa_utils import A_IsReplacedWith_B, preload_vqa_model
from prompt.guide import get_response, get_bot, system_prompt_gen_move_instructions, system_prompt_edit_sort
from basicsr.utils import tensor2img, img2tensor
from random import randint
from PIL import Image, ImageOps
import numpy as np
from detectron2.data import MetadataCatalog
from task_planning import Replace_Method, Move_Method, Transfer_Method
from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import calculate_clip_score, PSNR_compute, SSIM_compute
from preload_utils import *
from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial

def preload_replace_model(opt):
    return {
        'preloaded_example_generator': preload_example_generator(opt), 
        # XL - 8272 MiB, XL_ad - 8458 MiB, V1.5 - 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt),  # 8854 MiB
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt) # 10446 MiB
    }

def preload_move_model(opt):
    return {
        'preloaded_example_painter': preload_paint_by_example_model(opt), # 10446 MiB
        'preloaded_ip2p': preload_ip2p(opt),  # 8854 MiB
        'preloaded_sam_generator': preload_sam_generator(opt), # 10446 MiB
        'preloaded_seem_detector': preload_seem_detector(opt), # 10446 MiB
        'preloaded_lama_remover': preload_lama_remover(opt) # 10446 MiB
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

def write_valuation_results(path, clip_score=None, clip_directional_similarity=None, psnr_score=None, ssim_score=None, fid_score=None, extra_string=None):
    string = (f'Clip Score: {clip_score}\nClip Directional Similarity: {clip_directional_similarity}\n'
              f'PSNR: {psnr_score}\nSSIM: {ssim_score}\nFID: {fid_score}') + f"\n{extra_string}" if extra_string is not None else ""
    with open(path, 'w') as f:
        f.write(string)
    print(string)


def read_original_prompt(path_to_json):
    assert path_to_json.endswith('.json')
    with open(path_to_json, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
    prompt1 = data['input']
    edit = data['edit']
    prompt2 = f'{prompt1}, with {edit}'
    return (prompt1, prompt2, edit)

def Val_Replace_Method(opt):
    
    agent = use_exp_agent(opt, system_prompt_edit_sort)
    val_folder = '../autodl-tmp/COCO/val2017'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    with open('../autodl-tmp/COCO/annotations/instances_val2017.json') as f:
        data_val = json.load(f)['annotations']
        # query caption via image_id
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
            captions_dict[image_id] = captions_dict[image_id] + ', ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    image_before_list = image_after_list = image_ip2p_list = []
    caption_before_list = caption_after_list = []
    acc_num_replace = acc_num_ip2p = 0
    static_out_dir = opt.out_dir

    # 4-6 images in a folder
    while len(selected_list) < opt.test_group_num:
        start_time = time.time()
        idx = randint(0,length)
        while idx in selected_list:
            idx = randint(0, length)
        selected_list.append(idx)
        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.path.exists(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        # try:
        instance = data_val[idx]
        label_id = int(float(instance['category_id']))
        label_new_id = randint(1, 81)
        while label_new_id == label_id:
            label_new_id = randint(1, 81)
        label_ori = metadata.stuff_classes[label_id]
        label_new = metadata.stuff_classes[label_new_id]

        img_id = instance['image_id']
        img_path = os.path.join(val_folder, f'{img_id}:0{12}.jpg')

        ori_img = ImageOps.fit(Image.open(img_path).convert('RGB'), (256, 256), method=Image.Resampling.LANCZOS)
        opt.edit_txt = f'replace {label_ori} with {label_new}'
        caption1 = captions_dict[img_id]
        caption2 = f'{caption1}, with {label_ori} replaced with {label_new}'

        out_pil = Replace_Method(opt, 0, 0, ori_img, preloaded_replace_model, preloaded_agent, record_history=False)
        if out_pil.size != (256,256):
            out_pil = ImageOps.fit(out_pil.convert('RGB'), (256, 256), method=Image.Resampling.LANCZOS)
        out_ip2p = Transfer_Method(opt, 0, 0, ori_img, preloaded_replace_model, preloaded_agent, record_history=False)
        if out_ip2p.size != (256,256):
            out_ip2p = ImageOps.fit(out_ip2p.convert('RGB'), (256, 256), method=Image.Resampling.LANCZOS)

        ori_img.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
        out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
        out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-IP2P.jpg')
        write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', caption1, caption2, opt.edit_txt)

        image_before_list.append(ori_img)
        image_after_list.append(out_pil)
        image_ip2p_list.append(out_ip2p)
        caption_before_list.append(caption1)
        caption_after_list.append(caption2)

        a, b = A_IsReplacedWith_B(model_dict, label_ori, label_new, ori_img, [out_pil, out_ip2p], opt.device)
        acc_num_replace += a
        acc_num_ip2p += b

        end_time = time.time()

        string = (
            f'Images have been moved: {len(selected_list)} | Acc: [Add/Ip2p]~[{True if a == 1 else False}|'
            f'{True if b == 1 else False}] | Time cost: {end_time - start_time}')
        print(string)
        logging.info(string)

    # TODO: Clip Image Score & PSNR && SSIM

    # use list[Image]
    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    clip_directional_similarity_ip2p = cal_similarity(image_before_list, image_ip2p_list, caption_before_list, caption_after_list)

    fid_score = cal_fid(image_before_list, image_after_list)
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

    clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-base-patch16')
    clip_score = calculate_clip_score(image_after_list, caption_after_list, clip_score_fn=clip_score_fn)
    clip_score_ip2p = calculate_clip_score(image_ip2p_list, caption_after_list, clip_score_fn=clip_score_fn)
    del preloaded_agent, preloaded_replace_model

    acc_ratio_replace, acc_ratio_ip2p = acc_num_replace / len(selected_list), acc_num_ip2p / len(selected_list)
    # consider if there is need to save all images replaced
    string = f'Replace Acc: \n\tEditGPT = {acc_ratio_replace}\n\tIP2P = {acc_ratio_ip2p}\n'
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Remove.txt'), clip_score, clip_directional_similarity, psnr_score, ssim_score, fid_score, extra_string=string)
    write_valuation_results(os.path.join(static_out_dir, 'all_results_IP2P.txt'), clip_score_ip2p,
                            clip_directional_similarity_ip2p, psnr_score_ip2p, ssim_score_ip2p, fid_score_ip2p, extra_string=string)
    print(string)
    logging.info(string)

def Val_Move_Method(opt):
    
    val_folder = '../autodl-tmp/COCO/val2017/'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    agent = use_exp_agent(opt, system_prompt_gen_move_instructions)
    
    caption_before_list = caption_after_list = []
    image_before_list = image_after_list = image_ip2p_list = []

    # for validation after
    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        caption = json.load(f)    
    # query caption via image_id
    captions_dict = {}

    preloaded_move_model = preload_move_model(opt) if opt.preload_all_models else None
    preloaded_agent = preload_all_agents(opt) if opt.preload_all_agents else None

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
    
    static_out_dir = opt.out_dir

    while len(selected_list) < opt.test_group_num:
        while True:
            idx = randint(0, length)
            if idx in selected_list: continue
            else: break
        opt.out_dir = os.path.join(static_out_dir, f'{len(selected_list):0{6}}')
        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            selected_list.append(idx)
            annotation = data['annotations'][idx]
            start_time = time.time()
            
            x, y, w, h = annotation['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            img_id, label_id = annotation['image_id'], annotation['category_id']
            caption = captions_dict[str(img_id)]
            label = metadata.stuff_classes[int(float(label_id))]
            place = [x for x in get_response(agent, f'{caption}, {label}, {(x,y,w,h)}').split(';') if x != '' and x != ' ']
            assert len(place) == 2, f'place = {place}'
            ori_place, gen_place = place[0], place[1]
            
            img_path = os.path.join(val_folder, f'{img_id:0{12}}.jpg')
            img_pil = ImageOps.fit(Image.open(img_path).convert('RGB'), (256,256), method=Image.Resampling.LANCZOS)

            opt.edit_txt = f'move {label} from \'{ori_place}\' to \'{gen_place}\''
            out_pil = Move_Method(opt, 0, 0, img_pil, preloaded_move_model, preloaded_agent, record_history=False)
            if out_pil.size != (256,256):
                out_pil = ImageOps.fit(out_pil, (256,256), method=Image.Resampling.LANCZOS)
            out_ip2p = Transfer_Method(opt, 0, 0, img_pil, preloaded_move_model, preloaded_agent, record_history=False)
            if out_ip2p.size != (256,256):
                out_ip2p = ImageOps.fit(out_ip2p, (256,256), method=Image.Resampling.LANCZOS)

            end_time = time.time()
            string = f'Images have been moved: {len(selected_list)} | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

            image_before_list.append(img_pil)
            image_after_list.append(out_pil)
            image_ip2p_list.append(out_ip2p)
            c1, c2 = f'{label} at \'{ori_place}\'', f'{label} at \'{gen_place}\''
            caption_before_list.append(c1)
            caption_after_list.append(c2)

            img_pil.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output-EditGPT.jpg')
            out_ip2p.save(f'{opt.out_dir}/Inputs-Outputs/output-IP2P.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', c1, c2, opt.edit_txt)


        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]


    # TODO: Clip Image Score & PSNR && SSIM

    # use list[Image]
    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    clip_directional_similarity_ip2p = cal_similarity(image_before_list, image_ip2p_list, caption_before_list, caption_after_list)

    fid_score = cal_fid(image_before_list, image_after_list)
    fid_score_ip2p = cal_fid(image_before_list, image_ip2p_list)

    # use list[np.array]
    for i in range(len(image_after_list)):
        image_after_list[i] = np.array(image_after_list[i])
        image_before_list[i] = np.array(image_before_list[i])
        image_ip2p_list[i] = np.array(image_ip2p_list[i])

    clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-base-patch16')

    ssim_score = SSIM_compute(image_before_list, image_after_list)
    clip_score = calculate_clip_score(image_after_list, caption_after_list, clip_score_fn=clip_score_fn)
    psnr_score = PSNR_compute(image_before_list, image_after_list)

    ssim_score_ip2p = SSIM_compute(image_before_list, image_ip2p_list)
    clip_score_ip2p = calculate_clip_score(image_ip2p_list, caption_after_list, clip_score_fn=clip_score_fn)
    psnr_score_ip2p = PSNR_compute(image_before_list, image_ip2p_list)

    del preloaded_agent, preloaded_move_model
    # consider if there is need to save all images replaced
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Move.txt'), clip_score, clip_directional_similarity, psnr_score, ssim_score, fid_score)
    write_valuation_results(os.path.join(static_out_dir, 'all_results_Ip2p.txt'), clip_score_ip2p, clip_directional_similarity_ip2p, psnr_score_ip2p, ssim_score_ip2p, fid_score_ip2p)


def main():
    
    opt = get_arguments()
    setattr(opt, 'test_group_num', 100)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Replace-Move.log'
    )
    if os.path.isfile('Replace-Move.log'): os.system('Replace-Move.log')
    
    opt.out_dir = '../autodl-tmp/Exp_Replace'
    if os.path.exists(opt.out_dir): os.system(f'rm {opt.out_dir}.zip && zip {opt.out_dir}.zip {opt.out_dir} && rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Replace Method...')
    Val_Replace_Method(opt)
    
    opt.out_dir = '../autodl-tmp/Exp_Move'
    if os.path.exists(opt.out_dir): os.system(f'rm {opt.out_dir}.zip && zip {opt.out_dir}.zip {opt.out_dir} && rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Move Method...')
    Val_Move_Method(opt)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')