from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import cal_metrics_write
from PIL import Image
from basicsr import tensor2img, img2tensor
import numpy as np
from tqdm import tqdm
import pandas as pd


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

from operations.vqa_utils import *


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

from PIL import Image, ImageOps
from prompt.util import *
def main3():
    A = ImageOps.fit(Image.open('./assets/room.jpg').convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
    B = ImageOps.fit(Image.open('./assets/dog.jpg').convert('RGB'), (512,512), method=Image.Resampling.LANCZOS)
    A, B = np.array(A), np.array(B)
    
    psnr = PSNR_compute(A,B)
    print(psnr)

from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial
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

def Validate_planner():
    # validate the accuracy of GPT task planner on human labeled dataset
    base_path = '../autodl-tmp/planner-test-labeled/'
    all_data_folder = []
    big_folders = [os.path.join(base_path, x) for x in os.listdir(base_path) if 'engine' in x]
    for folder in big_folders:
        for gpt_i_folder in os.listdir(folder):
            # print(f'gpt_i_folder = {gpt_i_folder}')
            if 'zip' not in gpt_i_folder and '.DS_Store' not in gpt_i_folder:
                all_data_folder.append(os.path.join(folder, gpt_i_folder))
    raw_csv_list, ground_csv_list = [], []

    for folder in all_data_folder: # xxxx/GPT-1/
        # print(folder)
        # img_mapping = pd.read_csv(os.path.join(folder, 'data.csv'))
        raw_path = os.path.join(folder, 'GPT_gen_raw')
        for csv_ in os.listdir(raw_path):
            if not csv_.endswith('.csv'): continue
            raw_csv_list.append(os.path.join(raw_path, csv_))
        ground_path = os.path.join(folder, 'GPT_gen_label')
        for csv_ in os.listdir(ground_path):
            if not csv_.endswith('.csv'): continue
            ground_csv_list.append(os.path.join(ground_path, csv_))
    assert len(raw_csv_list) == len(ground_csv_list), f'len(raw_csv_list) = {len(raw_csv_list)}, len(ground_csv_list) = {len(ground_csv_list)}'
    print(f'len(raw_csv_list) = {len(raw_csv_list)}, len(ground_csv_list) = {len(ground_csv_list)}')

    from prompt.guide import get_response, get_bot, planning_system_prompt, planning_system_first_ask
    from task_planning import get_planns_directly

    engine = 'gpt-3.5-turbo'
    proxy = 'http://127.0.0.1'
    api_key = list(pd.read_csv('./key.csv')['key'])[0]

    planning_agent = get_bot(engine=engine, system_prompt=planning_system_prompt, api_key=api_key, proxy=proxy)
    _ = get_response(planning_agent, planning_system_first_ask)

    score_list = []
    tot = len(raw_csv_list)
    cur_dict = {}

    # tot = 1
    for i in range(tot):
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
        # cur_score =
        score_list.append(j / min(p, q))
        cur_dict[str(length)] = np.mean(score_list)

        print(f'Current Score [{(i+1):0{3}}|tot]: {cur_dict[str(length)]}')

    tot_score = np.mean(score_list)
    print(f'Test Planner: Average score-ratio on {len(score_list)} pieces data: {tot_score}')

    from raw_gen import csv_writer
    from matplotlib import pyplot as plt
    csv_writer('./dict-task-planning.csv', cur_dict)
    print(cur_dict)
    x, y = [], []
    for (k,v) in cur_dict.items():
        x.append(int(float(k)))
        y.append(v)
    plt.scatter(x, y, s=8, color='blue', marker='+')
    # while len(x) < 4:
    #     x.append(randint(1,10))
    #     y.append(randint(50,100)/100)
    # x, y = np.array(x), np.array(y)
    #
    # # Smooth Curve

    # from scipy.interpolate import interp1d
    # func = interp1d(x, y, kind='cubic')
    # x_smooth = np.linspace(np.min(x) - 1., np.max(x) + 1., 100)
    # plt.scatter(x_smooth, func(x_smooth), label='', color='blue')
    plt.xlabel('Task Length (ground truth)')
    plt.xticks(np.arange(min(np.min(x) - 1, 0), max(np.max(x) + 1, 12), 1))
    plt.ylabel('Accuracy')
    plt.title('task planning accuracy under our algorithm')
    # plt.legend()
    plt.grid(True)
    plt.savefig('./planner-curve.jpg')

    return tot_score, cur_dict
        # ⬇️ just counting
        # for cnt in range(0, min(len(label_list), len(plan_list))):
        #     if str(cnt) in acc_num_dict.keys():
        #         acc_num_dict[str(cnt)] += int(label_list[cnt] == plan_list[cnt])
        #     else:
        #         acc_num_dict[str(cnt)] = int(label_list[cnt] == plan_list[cnt])
        # for cnt in range(min(len(label_list), len(plan_list)), max(len(label_list), len(plan_list))):
        #     acc_num_dict[str(cnt)] = 0
        #
        # # acc_num_dict记录在序列索引为i的位置对应正确了几次
        # # TODO: 是一个分布列？
# done

def Figure_Multi_Plans():
    # draw figure: y[clip score, clip directional similarity, PSNR, SSIM] ~ x[number of plans]

    pass

if __name__ == '__main__':
    Validate_planner()