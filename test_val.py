from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import cal_metrics_write, PSNR_compute, SSIM_compute
from PIL import Image, ImageOps
from prompt.util import *
from basicsr import tensor2img, img2tensor
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging, json

from operations.vqa_utils import *
from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial

# validate multi task
from operations.utils import get_reshaped_img
from preload_utils import preload_all_models, preload_all_agents
from task_planning import get_operation_menu
from prompt.arguments import get_arguments

from prompt.guide import get_response, get_bot, planning_system_prompt, planning_system_first_ask
from task_planning import get_planns_directly


def main_1():
    c1 = 'a field'
    c2 = 'a field, with birds flying in the sky'

    img1 = Image.open('./assets/field.jpg').convert('RGB')
    img2 = Image.open('./Exp_plan/plans/plan01(add).jpg').convert('RGB')


    real_image_list = [img1]
    fake_image_list = [img2]

    caption_before_list = [c1]
    caption_after_list = [c2]

    clip_directional_similarity = cal_similarity(real_image_list, fake_image_list, caption_before_list, caption_after_list)
    print('clip directional similarity = ', clip_directional_similarity)

    fid_score = cal_fid(real_image_list, fake_image_list)
    print('fid score = ', fid_score)
# def main2():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model_base_path = '../autodl-tmp'
#     processor, model = preload(model_base_path, device)
#     ori = Image.open('./assets/flower.jpg')
#     remove = Image.open('../autodl-tmp/removed.jpg')
#     # replace = Image.open('../autodl-tmp/replaced.jpg')
#     model_dict = preload(model_base_path, device)
#     print(IsRemoved(model_dict, 'flowers', ori, remove, device))
#     # print(Val(model_dict, 'flowers',ori, ))

def main3():
    A = ImageOps.fit(Image.open('./assets/room.jpg').convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
    B = ImageOps.fit(Image.open('./assets/dog.jpg').convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
    A, B = np.array(A), np.array(B)
    
    psnr = PSNR_compute(A,B)
    print(psnr)

def main4():
    A = ImageOps.fit(Image.open('./input.jpg').convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
    B = ImageOps.fit(Image.open('./output.jpg').convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
    A, B = np.array(A), np.array(B)
    prompts = "A city street filled with lots of traffic and people.; "\
              "A busy street with cars, a motorcycle and a passenger bus; "\
              "A street with a motorcycle, bus and cars travelling on it. ; "\
              "Vehicles are traveling at both ends of the intersection with no "\
              "traffic control.; A bus, cars and a motorcycle driving in busy traffic on the street.; "\
              "with train replaced with suitcase. "
    prompts = prompts * 2
    clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-large-patch14')
    clip_score = calculate_clip_score([B] * 2, [prompts] * 2, clip_score_fn=clip_score_fn)
    print(clip_score)

def Validation_All():
    # For Interrupt Calculating
    for Name in ['Remove']:
    # for Name in ['Add', 'Remove', 'Replace', 'Move']:
        base_folder = f"../Exp_{Name}"
        # base_folder = f"../autodl-tmp/Exp_{Name}" # test move
        folders = os.listdir(base_folder)
        clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-large-patch14')
        in_img_list, EditGPT_img_list, Ip2p_img_list = [], [], []
        cap_1_list, cap_2_list = [], []
        cnt = 0

        for folder in folders:
            if not '0' in folder: continue
            test_folder = os.path.join(base_folder, folder, 'Inputs-Outputs')
            caption_path = os.path.join(test_folder, 'caption.txt')
            in_img_path = os.path.join(test_folder, 'input.jpg')
            EditGPT_img_path = os.path.join(test_folder, 'output-EditPGT.jpg')
            Ip2p_img_path = os.path.join(test_folder, 'output-Ip2p.jpg')
            with open(caption_path, 'r') as f:
                string = f.read().strip().split('\n')
            assert len(string) == 3

            cap_1_list.append(string[0])
            cap_2_list.append(string[1])
            in_img_list.append(Image.open(in_img_path))
            EditGPT_img_list.append(Image.open(EditGPT_img_path))
            Ip2p_img_list.append(Image.open(Ip2p_img_path))
            # print(f'{(len(in_img_list), len(EditGPT_img_list), len(Ip2p_img_list), len(cap_1_list), len(cap_2_list))}')
    
        cal_metrics_write(in_img_list, EditGPT_img_list, Ip2p_img_list, cap_1_list, cap_2_list, static_out_dir=base_folder, type_name=Name, extra_string=None)

def receive_from_csv(input_csv, type='raw'):
    if isinstance(input_csv, str):
        input_csv = pd.read_csv(input_csv)
    csv_dict = input_csv.to_dict(orient='records')
    assert len(csv_dict) == 1, f'csv_dict = \n\t{csv_dict}'
    csv_dict = csv_dict[0]
    # print(csv_dict)
    if type == 'raw':
        length = len(csv_dict)
        prompts = ''
        for (key, value) in csv_dict.items():
            # print(f'prompts = {prompts} |  csv_dict[key] = {csv_dict[key]}')
            prompts = prompts + ('' if prompts == '' else '; ') + csv_dict[key]
        return length, prompts
    elif type == 'label':
        return [value.lower().strip() for (_, value) in csv_dict.items()]


# def find_img_dict_file(csv_raw_path):
#     # csv_path: ......./GPT-x/xxxx.csv

def Validate_planner():
    opt = get_arguments()
    static_out_dir = opt.out_dir
    if os.path.exists(static_out_dir): os.system(f'rm -rf {static_out_dir}')
    os.mkdir(static_out_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename=f'{static_out_dir}/multi-task-valuation.log'
    )

    # prepare for COCO
    coco_base_path = '../autodl-tmp/COCO/val2017'
    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        caption = json.load(f)
    captions_dict = {}
    for x in caption['annotations']:
        image_id = str(x['image_id'])
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id] + '; ' + x['caption']
        else:
            captions_dict[image_id] = x['caption']

    # validate the accuracy of GPT task planner on human labeled dataset
    base_path = '../autodl-tmp/planner-test-labeled/'
    all_data_folder = []
    big_folders = [os.path.join(base_path, x) for x in os.listdir(base_path) if 'engine' in x]
    for folder in big_folders:
        for gpt_i_folder in os.listdir(folder):
            # print(f'gpt_i_folder = {gpt_i_folder}')
            if 'zip' not in gpt_i_folder and '.DS_Store' not in gpt_i_folder:
                all_data_folder.append(os.path.join(folder, gpt_i_folder))

    planning_agent = get_bot(engine=opt.engine, system_prompt=planning_system_prompt, api_key=opt.api_key, proxy=opt.proxy)
    _ = get_response(planning_agent, planning_system_first_ask)

    score_list = []
    cur_dict = {}

    operation_menu = get_operation_menu()
    preloaded_models = preload_all_models(opt) if opt.preload_all_models else None
    preloaded_agents = preload_all_agents(opt) if opt.prelaod_all_agents else None

    img_before, img_after, cap_before, cap_after = [], [], [], []
    cnt_global = 0

    for qwq in range(len(all_data_folder)): # xxxx/GPT-1/
        # print(folder)
        # img_mapping = pd.read_csv(os.path.join(folder, 'data.csv'))
        folder = all_data_folder[qwq]
        raw_path = os.path.join(folder, 'GPT_gen_raw')
        ground_path = os.path.join(folder, 'GPT_gen_label')
        image_path = os.path.join(folder, 'GPT_img')

        pre_read_coco_dict = pd.read_csv(os.path.join(folder, 'data.csv'))

        raw_csv_list = [os.path.join(raw_path, csv_) for csv_ in os.listdir(raw_path) if csv_.endswith('.csv')]
        ground_csv_list = [os.path.join(ground_path, csv_) for csv_ in os.listdir(ground_path) if csv_.endswith('.csv')]
        img_list = [os.path.join(image_path, img_) for img_ in os.listdir(image_path) if img_.endswith('.jpg')]

        assert len(raw_csv_list) == len(ground_csv_list), f'len(raw_csv_list) = {len(raw_csv_list)}, len(ground_csv_list) = {len(ground_csv_list)}'
        tot = len(raw_csv_list)

        tot = 1
        for i in range(tot):
            opt.out_dir = os.path.join(static_out_dir, f'{cnt_global:0{4}}')
            os.mkdir(opt.out_dir)
            cnt_global += 1
            length, prompts = receive_from_csv(raw_csv_list[i], type='raw')
            label_list = receive_from_csv(ground_csv_list[i], type='label')
            plans = get_planns_directly(planning_agent, prompts) # key: "type" in use | [{"type":..., "command":...}]
            plan_list = [x['type'].lower().strip() for x in plans]

            # Task Planner Validation Algorithm
            j, p, q = 0, len(plan_list), len(label_list)
            while j < min(p, q) and label_list[j] == plan_list[j]:
                j = j + 1
            if j == q - 1:
                j = j - (p - q)

            cur_score = j / min(p, q)
            score_list.append(cur_score)
            if str(length) in cur_dict.keys():
                cur_dict[str(length)].append(cur_score)
            else:
                cur_dict[str(length)] = [cur_score]

            score_string = f'Validate Step [{(i+1):0{3}}|{tot:0{3}}|{qwq:0{2}}]: current score: {cur_score}, average score: {np.mean(score_list)}'
            print(score_string)
            logging.info(score_string)

            planning_folder = os.path.join(opt.out_dir, 'plans')
            if not os.path.exists(planning_folder): os.mkdir(planning_folder)
            img_file = pre_read_coco_dict[str(int(img_list[i].strip('.jpg')))]
            opt.in_dir = os.path.join(coco_base_path, img_file)
            img_pil_before = get_reshaped_img(opt, img_pil=None)
            # cap1 = captions_dict[img_file.strip('.jpg')]
            # cap2 = f'{cap1}, edited by {cap}'

            for plan_item in plans:
                plan_type = plan_item['type']
                edit_tool = operation_menu[plan_type]
                opt.edit_txt = plan_item['command']

                img_pil_after, _ = edit_tool(
                        opt,
                        current_step = 0,
                        tot_step = 0,
                        input_pil = img_pil_before,
                        preloaded_model = preloaded_models,
                        preloaded_agent = preloaded_agents
                    )
                img_pil_before = img_pil_after

            img_before.append(img_pil_before)
            img_after.append(img_pil_after)

            # Token limitation: Fail to calculate CLIP related score
            # cap_before.append()
            # cap_after.append()

    tot_score = np.mean(score_list)
    print(f'Test Planner: Average score-ratio on {len(score_list)} pieces data: {tot_score}')



    from raw_gen import csv_writer
    from matplotlib import pyplot as plt
    csv_writer(f'{static_out_dir}/dict-task-planning.csv', cur_dict)
    print(cur_dict)
    x, y = [], []
    for (k,v) in cur_dict.items():
        x.append(int(float(k)))
        y.append(v)
    plt.plot(x, y, color='blue', marker='*')

    plt.xlabel('Task Length (ground truth)')
    plt.xticks(np.arange(min(np.min(x) - 1, 0), max(np.max(x) + 1, 12), 1))
    plt.ylabel('Accuracy')
    plt.title('Task Planning Accuracy Score')
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'{static_out_dir}/planner-curve.jpg')

    del preloaded_agents, preloaded_models

    ssim_score = SSIM_compute(img_before, img_after)
    psnr_score = PSNR_compute(img_before, img_after)
    write_valuation_results(os.path.join(base_path, 'multi-test-result.txt'), typer='Multi Task planning', clip_score='not cal', clip_directional_similarity='not cal', psnr_score=psnr_score, ssim_score=ssim_score)





def Figure_Multi_Plans():
    # draw figure: y[clip score, clip directional similarity, PSNR, SSIM] ~ x[number of plans]

    pass

if __name__ == '__main__':
    Validate_planner()