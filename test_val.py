from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from prompt.util import cal_metrics_write
from PIL import Image
from basicsr import tensor2img, img2tensor
import numpy as np
from Exp_replace_move import write_valuation_results

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


def main2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_base_path = '../autodl-tmp'
    processor, model = preload(model_base_path, device)
    ori = Image.open('./assets/flower.jpg')
    remove = Image.open('../autodl-tmp/removed.jpg')
    # replace = Image.open('../autodl-tmp/replaced.jpg')
    model_dict = preload(model_base_path, device)
    print(IsRemoved(model_dict, 'flowers', ori, remove, device))
    # print(Val(model_dict, 'flowers',ori, ))

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


def main5():
    base_folder = "../Exp_Remove" # test move
    folders = os.listdir(base_folder)
    clip_score_fn = partial(CLIP, model_name_or_path='../autodl-tmp/openai/clip-vit-large-patch14')
    in_img_list, EditGPT_img_list, Ip2p_img_list = [], [], []
    cap_1_list, cap_2_list = [], []
    
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
    
    
    cal_metrics_write(in_img_list, EditGPT_img_list, Ip2p_img_list, cap_1_list, cap_2_list, static_out_dir=base_folder, extra_string=None)
        
#     d_clip_EditGPT = cal_similarity(in_img_list, EditGPT_img_list, cap_1_list, cap_2_list)
#     d_clip_Ip2p = cal_similarity(in_img_list, Ip2p_img_list, cap_1_list, cap_2_list)

#     for i in range(len(in_img_list)):
#         in_img_list[i] = np.array(in_img_list[i])
#         EditGPT_img_list[i] = np.array(EditGPT_img_list[i])
#         Ip2p_img_list[i] = np.array(Ip2p_img_list[i])
    
    
#     clip_EditGPT = calculate_clip_score(EditGPT_img_list, cap_2)
#     clip_Ip2p = calculate_clip_score(Ip2p_img_list, cap_2)
    
#     ssim_EditGPT = SSIM_compute(in_img_list, EditGPT_img_list)
#     psnr_EditGPT = PSNR_compute(in_img_list, EditGPT_img_list)
    
#     ssim_Ip2p = SSIM_compute(in_img_list, Ip2p_img_list)
#     psnr_Ip2p = PSNR_compute(in_img_list, Ip2p_img_list)
    
#     write_valuation_results(os.path.join(base_folder, 'all_results_Remove_EditGPT.txt'), 'Remove-EditGPT', clip_EditGPT,
#                             d_clip_EditGPT, psnr_EditGPT, ssim_EditGPT, 0)
#     write_valuation_results(os.path.join(base_folder, 'all_results_Remove.txt'), 'Remove-EditIp2p', clip_Ip2p,
#                             d_clip_Ip2p, psnr_Ip2p, ssim_Ip2p, 0)
    


if __name__ == '__main__':
    main5()