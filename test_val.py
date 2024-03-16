from prompt.util import Cal_ClipDirectionalSimilarity as cal_similarity
from prompt.util import Cal_FIDScore as cal_fid
from PIL import Image
from basicsr import tensor2img, img2tensor
import numpy as np

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




if __name__ == '__main__':
    main2()