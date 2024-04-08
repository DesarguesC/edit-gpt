import os, time, json, logging

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
from task_planning import get_operation_menu, get_planning_system_agent, get_plans_directly, get_plans

from prompt.arguments import get_arguments
from prompt.util import calculate_clip_score, PSNR_compute, SSIM_compute, write_instruction, write_valuation_results, cal_metrics_write
from preload_utils import *
from torchmetrics.functional.multimodal import clip_score as CLIP
from functools import partial
from pytorch_lightning import seed_everything
from operations.utils import get_reshaped_img
from preload_utils import preload_all_models, preload_all_agents
from task_planning import get_operation_menu, get_planning_system_agent, get_plans_directly, get_plans
from prompt.arguments import get_arguments

from prompt.guide import get_response, get_bot, planning_system_prompt, planning_system_first_ask

from socket import *
import numpy as np
import cv2, time
from PIL import Image, ImageOps
from tcputils import receive_image_from_length, Encode_and_Send

def Validate_on_IPr2IPr(opt, preloaded_models, preloaded_agents, test_num=50, clientSocket=None):

    seed_everything(opt.seed)
    if not hasattr(opt, 'test_num'): setattr(opt, 'test_num', test_num)
    # draw figure: y[clip score, clip directional similarity, PSNR, SSIM] ~ x[number of plans]
    dataset_path = '../autodl-tmp/clip-filtered/shard-00'
    folders = os.listdir(dataset_path)
    operation_menu = get_operation_menu()

    img_before_list, img_after_list = [], [] # editgpt
    SDEdit_after_list , Ip2p_after_list, MGIE_after_list = [], [], []
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
            opt, img_pil = get_reshaped_img(opt, val=True, val_shape=(512,512))
            img_before = img_pil

            # evaluate other similar model first
            if opt.with_ip2p_val:
                opt.model_type = 'IP2P'
                img_pil_ip2p = Transfer_Method(opt, 0, 0, img_pil, preloaded_models, preloaded_agents,
                                       record_history=False, model_type=opt.model_type, clientSocket=None,
                                       size=(512, 512))
                opt.model_type = 'SDEdit'
                img_pil_sdedit = Transfer_Method(opt, 0, 0, img_pil, preloaded_models, preloaded_agents,
                                       record_history=False, model_type=opt.model_type, clientSocket=None,
                                       size=(512, 512))
                opt.model_type = 'MGIE'
                img_pil_mgie = Transfer_Method(opt, 0, 0, img_pil, preloaded_models, preloaded_agents,
                                       record_history=False, model_type=opt.model_type, clientSocket=clientSocket,
                                       size=(512, 512))


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

            if img_pil.size != (512, 512):
                img_pil = ImageOps.fit(img_pil.convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)

            img_before_list.append(img_before)
            img_after_list.append(img_pil)
            cap_before_list.append(data['input'])
            cap_after_list.append(data['output'])

            if opt.with_ip2p_val:
                Ip2p_after_list.append(img_pil_ip2p)
                SDEdit_after_list.append(img_pil_sdedit)
                MGIE_after_list.append(img_pil_mgie)

        if cnt > opt.test_num: break

    cal_metrics_write(
        img_before_list, img_after_list,
        None, cap_before_list,
        cap_after_list, static_out_dir=static_out_dir,
        type_name=' - Val All - ', extra_string="", model_type='EditGPT'
    )

    if opt.with_ip2p_val:
        cal_metrics_write(
            img_before_list, Ip2p_after_list,
            None, cap_before_list,
            cap_after_list, static_out_dir=static_out_dir,
            type_name=' - Val All - ', extra_string="", model_type='IP2P'
        )
        cal_metrics_write(
            img_before_list, SDEdit_after_list,
            None, cap_before_list,
            cap_after_list, static_out_dir=static_out_dir,
            type_name=' - Val All - ', extra_string="", model_type='SDEdit'
        )
        cal_metrics_write(
            img_before_list, SDEdit_after_list,
            None, cap_before_list,
            cap_after_list, static_out_dir=static_out_dir,
            type_name=' - Val All - ', extra_string="", model_type='MGIE'
        )


if __name__ == '__main__':
    start_time = time.time()
    opt = get_arguments()
    preloaded_models = preload_all_models(opt)
    preloaded_agents = preload_all_agents(opt)

    clientSocket = None
    if opt.with_ip2p_val:
        clientHost, clientPort = '127.0.0.1', 4096
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((clientHost, clientPort))

    Validate_on_IPr2IPr(opt, preloaded_models, preloaded_agents, test_num=1, clientSocket=clientSocket)
    end_time = time.time()
    print(f'Total Main func, Valuation cost: {end_time - start_time} (seconds).')