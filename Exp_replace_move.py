import os, time, json, logging
from prompt.guide import get_response, get_bot, system_prompt_gen_move_instructions, system_prompt_edit_sort
from basicsr.utils import tensor2img, img2tensor
from random import randint
from PIL import Image, ImageOps
import numpy as np
from task_planning import Replace_Method, Move_Method
from prompt.arguments import get_arguments
from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import calculate_clip_score, PSNR_compute, SSIM_compute
from detectron2.data import MetadataCatalog
from preload_utils import *
from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial

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

def write_valuation_results(path, clip_score=None, clip_directional_similarity=None, psnr_score=None, ssim_score=None, fid_score=None):
    string = (f'Clip Score: {clip_score}\nClip Directional Similarity: {clip_directional_similarity}\n'
              f'PSNR: {psnr_score}\nSSIM: {ssim_score}\nFID: {fid_score}')
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

    static_out_dir = opt.out_dir

    # 4-6 images in a folder
    while len(executed_list) < opt.test_group_num:
        opt.out_dir = os.path.join(static_out_dir, f'{len(executed_list):0{6}}')
        idx = randint(0, length)
        while idx in selected_list: idx = randint(0, length)
        selected_list.append(idx)

        folder = folders[idx]
        work_folder = os.path.join(val_folder, folder)
        json_path = os.path.join(work_folder, 'prompt.json')
        c1, c2, edit = read_original_prompt(json_path)
        sorted = get_response(agent, edit)
        if not 'replace' in sorted.lower(): continue
        else: executed_list.append(idx)

        if not os.path.exists(opt.out_dir):
            os.mkdir(opt.out_dir)
            os.mkdir(f'{opt.out_dir}/Inputs-Outputs/')

        try:
            name_list = [img.split('_')[0] for img in os.listdir(work_folder) if img.endswith('.jpg')]
            name_list = list(set(name_list))
            opt.edit_txt = edit
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/captions.txt', c1, c2, edit)
            for ii in range(len(name_list)):
                name = name_list[ii]
                start_time = time.time()

                img_path = os.path.join(work_folder, f'{name}_0.jpg')
                img_pil = ImageOps.fit(Image.open(img_path).convert('RGB'), (256,256), method=Image.Resampling.LANCZOS)

                output_pil = Replace_Method(opt, 0, 0, img_pil, preloaded_replace_model, preloaded_agent, record_history=False)
                output_pil = ImageOps.fit(output_pil, (256,256), method=Image.Resampling.LANCZOS)
                img_pil.save(f'{opt.out_dir}/Inputs-Outputs/inputs-{ii}.jpg')
                output_pil.save(f'{opt.out_dir}/Inputs-Outputs/outputs-{ii}.jpg')

                caption_before_list.append(c1)
                caption_after_list.append(c2)
                fake_image_list.append(output_pil) # pil list
                real_fake_image_list.append(img_pil) # pil list
                # use _0 or _1 ?
                execute_img_cnt += 1

                end_time = time.time()

                string = f'Images have been Replaced: {execute_img_cnt} | Time Cost: {end_time - start_time}'
                print(string)
                logging.info(string)

        except Exception as e:
            string = f'Exceptino Occurred: {e}'
            print(string)
            logging.error(string)
            del executed_list[-1]

    # TODO: Clip Image Score & PSNR && SSIM

    # use list[Image]
    clip_directional_similarity = cal_similarity(real_fake_image_list, fake_image_list, caption_before_list, caption_after_list)
    fid_score = cal_fid(real_fake_image_list, fake_image_list)

    # use list[np.array]
    for i in range(len(fake_image_list)):
        fake_image_list[i] = np.array(fake_image_list[i])
        real_fake_image_list[i] = np.array(real_fake_image_list[i])

    ssim_score = SSIM_compute(real_fake_image_list, fake_image_list)
    clip_score = calculate_clip_score(fake_image_list, caption_after_list, clip_score_fn=partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-base-patch16'))
    psnr_score = PSNR_compute(real_fake_image_list, fake_image_list)

    
    del preloaded_agent, preloaded_replace_model
    # consider if there is need to save all images replaced
    write_valuation_results(os.path.join(static_out_dir, 'all_results.txt'), clip_score, clip_directional_similarity, psnr_score, ssim_score, fid_score)

def Val_Move_Method(opt):
    
    val_folder = '../autodl-tmp/COCO/val2017/'
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    agent = use_exp_agent(opt, system_prompt_gen_move_instructions)
    
    caption_before_list = caption_after_list = []
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
            out_pil = ImageOps.fit(out_pil, (256,256), method=Image.Resampling.LANCZOS)
            
            end_time = time.time()
            string = f'Images have been moved: {len(selected_list)} | Time cost: {end_time - start_time}'
            print(string)
            logging.info(string)

            image_before_list.append(img_pil)
            image_after_list.append(out_pil)
            c1, c2 = f'{label} at \'{ori_place}\'', f'{label} at \'{gen_place}\''
            caption_before_list.append(c1)
            caption_after_list.append(c2)

            img_pil.save(f'{opt.out_dir}/Inputs-Outputs/input.jpg')
            out_pil.save(f'{opt.out_dir}/Inputs-Outputs/output.jpg')
            write_instruction(f'{opt.out_dir}/Inputs-Outputs/caption.txt', c1, c2, opt.edit_txt)


        except Exception as e:
            string = f'Exception Occurred: {e}'
            print(string)
            logging.error(string)
            del selected_list[-1]


    # TODO: Clip Image Score & PSNR && SSIM

    # use list[Image]
    clip_directional_similarity = cal_similarity(image_before_list, image_after_list, caption_before_list, caption_after_list)
    fid_score = cal_fid(image_before_list, image_after_list)

    # use list[np.array]
    for i in range(len(image_after_list)):
        image_after_list[i] = np.array(image_after_list[i])
        image_before_list[i] = np.array(image_before_list[i])

    ssim_score = SSIM_compute(image_before_list, image_after_list)
    clip_score = calculate_clip_score(image_after_list, caption_after_list, clip_score_fn=partial(CLIP,
                                                                                                 model_name_or_path='../autodl-tmp/openai/clip-vit-base-patch16'))
    psnr_score = PSNR_compute(image_before_list, image_after_list)

    del preloaded_agent, preloaded_move_model
    # consider if there is need to save all images replaced
    write_valuation_results(os.path.join(static_out_dir, 'all_results.txt'), clip_score, clip_directional_similarity, psnr_score, ssim_score, fid_score)


def main():
    
    opt = get_arguments()
    setattr(opt, 'test_group_num', 100)

    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s : %(levelname)s : %(message)s', 
        filename='Replace&Move.log'
    )
    if os.path.isfile('Replace-Move.log'): os.system('Replace-Move.log')
    
    opt.out_dir = '../autodl-tmp/Exp_Replace/'
    if os.path.exists(opt.out_dir): os.system(f'rm {opt.out_dir}.zip && zip {opt.out_dir}.zip {opt.out_dir} && rm -rf {opt.out_dir}')
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    base_cnt = len(os.listdir(opt.out_dir))
    setattr(opt, 'base_cnt', base_cnt)
    print('Start to valuate Replace Method...')
    Val_Replace_Method(opt)
    
    opt.out_dir = '../autodl-tmp/Exp_Move/'
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