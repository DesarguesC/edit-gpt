from operations import Remove_Me, Remove_Me_lama
from seem.masks import middleware
from paint.crutils import ab8, ab64
from paint.bgutils import refactor_mask, match_sam_box
from paint.example import paint_by_example
from PIL import Image
import numpy as np
# from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog as mt
from torch import autocast
from torch.nn import functional as F
from prompt.item import Label
from prompt.guide import get_response, get_bot, system_prompt_expand
from jieba import re
from seem.masks import middleware, query_middleware
from ldm.inference_base import *
from paint.control import *
from basicsr.utils import tensor2img, img2tensor
from pytorch_lightning import seed_everything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
from einops import repeat, rearrange
import torch, cv2
from paint.example import generate_example



def Add_Object(opt, name: str, num: int, place: str, edit_agent=None, expand_agent=None):
    assert edit_agent != None, 'no edit agent!'
    img_pil = Image.open(opt.in_dir).convert('RGB')
    opt.W, opt.H = img_pil.size
    opt.W, opt.H = ab64(opt.W), ab64(opt.H)
    img_pil = img_pil.resize((opt.W, opt.H))

    if '<NULL>' in place:
        # system_prompt_add -> panoptic
        _, panopic_dict = middleware(opt, img_pil)
        question = Label().get_str_add_panoptic(panopic_dict, name, (opt.W,opt.H))
    else:
        # system_prompt_addArrange -> name2place
        _, place_mask, place_box = query_middleware(opt, img_pil, place)
        question = Label().get_str_add_place(place, name, (opt.W,opt.H), place_box)

    for i in range(num):
        ans = get_response(edit_agent, question)
        ans_list = [x.strip() for x in re.split('[(),{}]', ans) if x != '' and x != ' ']
        assert len(ans_list) == 5, f'ans = {ans}, ans_list = {ans_list}'
        ori_box = (int(ans_list[1]), int(ans_list[2]), int(ans_list[3]), int(ans_list[4]))
        # generate example
        diffusion_pil = generate_example(opt, name, expand_agent=expand_agent, use_inpaint_adapter=opt.use_inpaint_adpter, \
                                                        ori_img=img_pil)
        # query mask-box & rescale
        _, mask_example, _ = query_middleware(opt, diffusion_pil, name)
        sam = sam_model_registry[opt.sam_type](checkpoint=opt.sam_ckpt)
        sam.to(device=opt.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        box_example = match_sam_box(mask_2, [
            (u['bbox'], u['segmentation'], u['area']) for u in \
                    sorted(mask_generator.generate(cv2.cvtColor(np.array(diffusion_pil), cv2.COLOR_RGB2BGR)), \
                                           key=(lambda x: x['area']), reverse=True)
                    ])

        target_mask = refactor_mask(box_example, mask_example, ori_box)
        # paint-by-example
        _, painted = paint_by_example(opt, mask=target_mask, ref_img=diffusion_pil, \
                                   base_img=img_pil, use_adapter=opt.use_pbe_adapter)
        output_path = os.path.join(opt.out_dir, f'{i}~{opt.out_name}')
        cv2.imwrite(output_path, tensor2img(painted))
        print(f'Added image saved at \'{output_path}\'')

    print('exit from add')
    exit(0)






