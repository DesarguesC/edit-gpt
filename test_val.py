import os.path

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
from task_planning import get_operation_menu, get_planning_system_agent, get_planns_directly, get_plans
from prompt.arguments import get_arguments

from prompt.guide import get_response, get_bot, planning_system_prompt, planning_system_first_ask



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
        return [str(value).lower().strip() for (_, value) in csv_dict.items()]


# def find_img_dict_file(csv_raw_path):
#     # csv_path: ......./GPT-x/xxxx.csv

def Validate_planner():
    opt = get_arguments()
    static_out_dir = opt.out_dir
    if not os.path.exists(static_out_dir):
        os.mkdir(static_out_dir) # os.system(f'rm -rf {static_out_dir}')

    if os.path.isfile(os.path.join(static_out_dir, 'multi-task-valuation.log')): os.system('rm multi-task-valuation.log')
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

    score_list = []
    cur_dict = {}

    operation_menu = get_operation_menu()
    preloaded_models = preload_all_models(opt) if opt.preload_all_models else None
    preloaded_agents = preload_all_agents(opt) if opt.preload_all_agents else None
    planning_agent = get_planning_system_agent(opt)

    img_before, img_after, cap_before, cap_after = [], [], [], []
    img_dict = {} # cap_before_dict, cap_after_dict = {}, {}, {}, {}
    print(f'len(os.listdir(static_out_dir)) = {len(os.listdir(static_out_dir))}')

    cnt_global = len([x for x in os.listdir(static_out_dir) if not (os.path.isfile(x) or x in ['.', '..', '.ipynb_checkpoints'])])

    print(f'\nAll Data Folder amount: {len(all_data_folder)}\nGlobal cnt = {cnt_global}')

    for qwq in range(len(all_data_folder)): # xxxx/GPT-1/
        qwq = 3
        # TODO: Only a folder one time !
        folder = all_data_folder[qwq]

        raw_path = os.path.join(folder, 'GPT_gen_raw')
        ground_path = os.path.join(folder, 'GPT_gen_label')
        image_path = os.path.join(folder, 'GPT_img')

        pre_read_coco_dict = pd.read_csv(os.path.join(folder, 'data.csv'))

        raw_csv_list = [os.path.join(raw_path, csv_) for csv_ in os.listdir(raw_path) if csv_.endswith('.csv')]
        ground_csv_list = [os.path.join(ground_path, csv_) for csv_ in os.listdir(ground_path) if csv_.endswith('.csv')]
        raw_img_list = [img_ for img_ in os.listdir(image_path) if img_.endswith('.jpg')]
        # img_list = [os.path.join(image_path, img_) for img_ in raw_img_list]

        assert len(raw_csv_list) == len(ground_csv_list), f'len(raw_csv_list) = {len(raw_csv_list)}, len(ground_csv_list) = {len(ground_csv_list)}'
        tot = len(raw_csv_list)

        for i in range(tot):
            # if i > randint(1,3): break
            # try:
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

            score_string = f'\n\tValidate Step [{(i+1):0{3}}|{tot:0{3}}]|[{(qwq+1):0{2}}|{(len(all_data_folder)+1):0{2}}] †† current score: {cur_score}, average score: {np.mean(score_list)}\n'
            print(score_string)
            logging.info(score_string)


            # Starts Editing Images according to Plans
            planning_folder = os.path.join(opt.out_dir, 'plans')
            if not os.path.exists(planning_folder): os.mkdir(planning_folder)
            # print(pre_read_coco_dict)
            img_file = list(pre_read_coco_dict[str(raw_img_list[i].strip('.jpg'))])[0] # ends with '.jpg'
            opt.in_dir = os.path.join(coco_base_path, img_file)
            opt, img_pil_before = get_reshaped_img(opt, img_pil=None, val=True) # opt return, obtain (W, H)
            io_path = os.path.join(opt.out_dir, 'Input-Output')
            if not os.path.exists(io_path): os.mkdir(io_path)

            img_pil_before.save(f'{io_path}/input.jpg')
            print(f'img_file (id) = {img_file}')
            # print(f'captions_dict.keys() = {captions_dict.keys()}')
            cap1 = captions_dict[img_file.strip('.jpg').lstrip('0')]
            cap2 = f'{cap1}, edited by \'{prompts}\''
            with open(os.path.join(io_path, 'caption.txt'), 'w') as f:
                f.write(f'{cap1}\n\n{cap2}\n\n{prompts}')


            plan_step, tot_step = 1, len(plans)

            for plan_item in plans:
                plan_type = plan_item['type']
                edit_tool = operation_menu[plan_type]
                opt.edit_txt = plan_item['command']

                img_pil_after, _ = edit_tool(
                        opt,
                        current_step = plan_step,
                        tot_step = tot_step,
                        input_pil = img_pil_before,
                        preloaded_model = preloaded_models,
                        preloaded_agent = preloaded_agents
                    )
                img_pil_before = img_pil_after

                img_pil_after.save(f'./{planning_folder}/plan{plan_step:02}({plan_type}).jpg')
                plan_step += 1
            
            img_before.append(img_pil_before)
            img_after.append(img_pil_after)
            img_pil_before.save(f'{io_path}/output.jpg')

            if str(length) in img_dict.keys():
                img_dict[str(length)][0].append(img_pil_before)
                img_dict[str(length)][1].append(img_pil_after)
            else:
                img_dict[str(length)] = [[img_pil_before], [img_pil_after]]


            # except Exception as err:
            #     string = f'Error occurred: {err}'
            #     print(string)
            #     logging.info(string)

            # Token limitation: Fail to calculate CLIP related score
            # cap_before.append()
            # cap_after.append()
        with open(os.path.join(static_out_dir, f'score-dict-{qwq}.txt'), 'w') as f:
            f.write(str(cur_dict))

        break # TODO: Only a folder at one time.

    tot_score = np.mean(score_list)
    print(f'Test Planner: Average score-ratio on {len(score_list)} pieces data: {tot_score}')

    from raw_gen import csv_writer
    from matplotlib import pyplot as plt
    csv_writer(f'{static_out_dir}/dict-task-planning.csv', cur_dict)
    # print(cur_dict)
    x, y = [], []
    for (k,v) in cur_dict.items():
        x.append(int(float(k)))
        y.append(np.mean(v))
    
    assert len(x) == len(y), f'len(x) = {len(x)}, len(y) = {y}'
    plt.scatter(np.array(x), np.array(y), color='blue', marker='*')
    plt.plot(x, y, 'b-')

    plt.xlabel('Task Length (ground truth)')
    plt.xticks(np.arange(min(np.min(x) - 1, 0), max(np.max(x) + 1, 12), 1))
    plt.ylabel('Accuracy')
    plt.title('Task Planning Accuracy Score')
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'{static_out_dir}/planner-curve.jpg')
    
    del preloaded_agents, preloaded_models
    for i in range(len(img_before)):
        img_before[i] = np.array(img_before[i])
        img_after[i] = np.array(img_after[i])

    ssim_score = SSIM_compute(img_before, img_after)
    psnr_score = PSNR_compute(img_before, img_after)
    extra_string = f'tot task planning accuracy = {tot_score}'
    writing_string = write_valuation_results(os.path.join(base_path, 'multi-test-result.txt'), typer='Multi Task planning', psnr_score=psnr_score, ssim_score=ssim_score)
    logging.info(writing_string)
    
    psnr_dict, ssim_dict = {}, {}
    index, ssim_list, psnr_list = [], [], []
    for (k, v) in img_dict.items():
        index.append(int(k))
        for i in range(len(v[0])):
            v[0][i] = np.array(v[0][i])
            v[1][i] = np.array(v[0][i])
         
        ssim_dict[k] = SSIM_compute(v[0], v[1])
        psnr_dict[k] = PSNR_compute(v[0], v[1])
            
        ssim_list.append(psnr_dict[k])
        psnr_list.append(ssim_dict[k])
    
    plt.scatter(np.array(index), np.array(psnr_list), color='blue', marker='*')
    plt.scatter(np.array(index), np.array(ssim_list), color='red', marker='+')
    
    plt.plot(index, psnr_list, 'b-', label='PSNR')
    plt.plot(index, ssim_list, 'r-', label='SSIM')
    
    plt.legend()
    plt.xticks(np.arange(min(np.min(index) - 1,  0), max(np.max(index) + 1, 12), 1))
    plt.xlabel('Task Steps')
    plt.ylabel('Scores')
    plt.savefig(f'{static_out_dir}/planner-score-curve.jpg')
    
    csv_writer(f'{static_out_dir}/psnr-dict-task-{qwq}.csv', psnr_dict)
    csv_writer(f'{static_out_dir}/ssim-dict-task-{qwq}.csv', ssim_dict)


def Validate_planner_No_Img():
    opt = get_arguments()
    static_out_dir = opt.out_dir
    if not os.path.exists(static_out_dir):
        os.mkdir(static_out_dir)  # os.system(f'rm -rf {static_out_dir}')

    if os.path.isfile(os.path.join(static_out_dir, 'multi-task-valuation.log')): os.system(
        'rm multi-task-valuation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename=f'{static_out_dir}/multi-task-valuation.log'
    )

    # validate the accuracy of GPT task planner on human labeled dataset
    base_path = '../autodl-tmp/planner-test-labeled/'
    all_data_folder = []
    big_folders = [os.path.join(base_path, x) for x in os.listdir(base_path) if 'engine' in x]
    for folder in big_folders:
        for gpt_i_folder in os.listdir(folder):
            if 'zip' not in gpt_i_folder and '.DS_Store' not in gpt_i_folder:
                all_data_folder.append(os.path.join(folder, gpt_i_folder))

    score_list = []
    cur_dict = {}
    planning_agent = get_planning_system_agent(opt)
    print(f'len(os.listdir(static_out_dir)) = {len(os.listdir(static_out_dir))}')

    cnt_global = len(
        [x for x in os.listdir(static_out_dir) if not (os.path.isfile(x) or x in ['.', '..', '.ipynb_checkpoints'])])

    print(f'\nAll Data Folder amount: {len(all_data_folder)}\nGlobal cnt = {cnt_global}')

    for qwq in range(len(all_data_folder)):  # xxxx/GPT-1/
        # TODO: Only a folder one time !
        # if qwq < 6: continue
        folder = all_data_folder[qwq]

        raw_path = os.path.join(folder, 'GPT_gen_raw')
        ground_path = os.path.join(folder, 'GPT_gen_label')

        raw_csv_list = [os.path.join(raw_path, csv_) for csv_ in os.listdir(raw_path) if csv_.endswith('.csv')]
        ground_csv_list = [os.path.join(ground_path, csv_) for csv_ in os.listdir(ground_path) if csv_.endswith('.csv')]

        assert len(raw_csv_list) == len(
            ground_csv_list), f'len(raw_csv_list) = {len(raw_csv_list)}, len(ground_csv_list) = {len(ground_csv_list)}'
        tot = len(raw_csv_list)
        print(f'tot = {tot}')
        # tot = 20
        for i in range(tot):
            opt.out_dir = os.path.join(static_out_dir, f'{cnt_global:0{4}}')
            os.mkdir(opt.out_dir)
            cnt_global += 1
            # print(f'raw_csv_list[i] = {raw_csv_list[i]}')
            # print(f'ground_csv_list[i] = {ground_csv_list[i]}')
            length, prompts = receive_from_csv(raw_csv_list[i], type='raw')
            label_list = receive_from_csv(ground_csv_list[i], type='label')
            plans = get_planns_directly(planning_agent, prompts)  # key: "type" in use | [{"type":..., "command":...}]
            plan_list = [x['type'].lower().strip() for x in plans]
            # print(f'length = {length}, prompts = {prompts}')

            # Task Planner Validation Algorithm
            j, p, q = 0, len(plan_list), len(label_list)
            while j < min(p, q) and label_list[j] == plan_list[j]:
                j = j + 1
            if j == q - 1:
                j = j - (p - q)

            # assert min(p, q) != 0, f'csv: raw: {raw_csv_list[i]}, ground: {ground_csv_list[i]}'
            if min(p,q) == 0:
                cur_score = 0.
            else:
                cur_score = j / min(p, q)
            score_list.append(cur_score)
            if str(length) in cur_dict.keys():
                cur_dict[str(length)].append(cur_score)
            else:
                cur_dict[str(length)] = [cur_score]

            score_string = f'\n\tValidate Step [{(i + 1):0{3}}|{tot:0{3}}]|[{(qwq + 1):0{2}}|{(len(all_data_folder) + 1):0{2}}] †† current score: {cur_score}, average score: {np.mean(score_list)}\n'
            print(score_string)
            logging.info(score_string)

    tot_score = np.mean(score_list)
    print(f'Test Planner: Average score-ratio on {len(score_list)} pieces data: {tot_score}')

    from raw_gen import csv_writer
    from matplotlib import pyplot as plt
    csv_writer(f'{static_out_dir}/dict-task-planning.csv', cur_dict)
    # print(cur_dict)
    x, y = [], []
    for (k, v) in cur_dict.items():
        x.append(int(float(k)))
        y.append(np.mean(v))

    assert len(x) == len(y), f'len(x) = {len(x)}, len(y) = {y}'
    plt.scatter(np.array(x), np.array(y), color='blue', marker='*')
    plt.plot(x, y, 'b-')

    plt.xlabel('Task Length (ground truth)')
    plt.xticks(np.arange(min(np.min(x) - 1, 0), max(np.max(x) + 1, 12), 1))
    plt.ylabel('Accuracy')
    plt.title('Task Planning Accuracy Score')
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'{static_out_dir}/planner-curve.jpg')


def Validate_on_Ip2p_Dataset(test_num):
    opt = get_arguments()
    if not hasattr(opt, 'test_num'): setattr(opt, 'test_num', test_num)
    preloaded_models = preload_all_models(opt)
    preloaded_agents = preload_all_agents(opt)
    # draw figure: y[clip score, clip directional similarity, PSNR, SSIM] ~ x[number of plans]
    dataset_path = '../autodl-tmp/clip-filtered/shard-00'
    folders = os.listdir(dataset_path)
    operation_menu = get_operation_menu()

    img_before_list, img_after_list = [], []
    cap_before_list, cap_after_list = [], []
    cnt = 0
    static_out_dir = opt.out_dir
    if not os.path.exists(static_out_dir): os.mkdir(static_out_dir)

    planning_agent = get_planning_system_agent(opt)

    for folder in folders:

        file_list = os.listdir(os.path.join(dataset_path, folder))
        data = json.load(os.path.join(dataset_path, folder, 'prompt.json'))
        opt.edit_txt = data['edit']
        task_plannings = get_plans(opt, planning_agent)

        img_name_list = list(set([x.split('_')[0] for x in file_list if x.endswith('.jpg')]))

        for img_name in img_name_list:
            cnt = cnt + 1
            opt.out_dir = os.path.join(static_out_dir, f'{cnt:0{3}}')
            os.mkdir(opt.out_dir)
            planning_folder = os.path.join(opt.out_dir, 'plans')
            if not os.path.exists(planning_folder): os.mkdir(planning_folder)
            plan_step, tot_step = 1, len(task_plannings)

            opt.in_dir = os.path.join(dataset_path, folder, f'{img_name}_0.jpg')
            opt, img_pil = get_reshaped_img(opt)
            img_before = img_pil

            for plan_item in task_plannings:
                plan_type = plan_item['type']
                edit_tool = operation_menu[plan_type]
                opt.edit_txt = plan_item['command']

                img_pil, _ = edit_tool(
                        opt,
                        current_step = plan_step,
                        tot_step = tot_step,
                        input_pil = img_pil,
                        preloaded_model = preloaded_models,
                        preloaded_agent = preloaded_agents
                    )
                img_pil.save(f'./{planning_folder}/plan{plan_step:02}({plan_type}).jpg')
                plan_step += 1

            img_before_list.append(img_before)
            img_after_list.append(img_pil)
            cap_before_list.append(data['input'])
            cap_after_list.append(data['output'])

        if cnt > opt.test_num: break

    psnr_score = PSNR_compute(img_before_list, img_after_list)
    ssim_score = SSIM_compute(img_before_list, img_after_list)

    writing_string = write_valuation_results(os.path.join(dataset_path, 'test-on-ip2p-dataset.txt'),
                                             typer='Multi Task planning', psnr_score=psnr_score, ssim_score=ssim_score)
    logging.info(writing_string)
    


if __name__ == '__main__':
    Validate_planner_No_Img()