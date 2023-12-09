import argparse
from prompt.crfill_init import initialize
from options.base_options import BaseOptions
from options.test_options import TestOptions
from ldm.inference_base import get_base_argument_parser

def get_args():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    
    # parser = BaseOptions().initialize(parser)
    parser = TestOptions().initialize(parser)
    parser = get_base_argument_parser(parser)
    
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", help='path to config file', )
    # set as default
    parser.add_argument('--in_dir', default='../autodl-tmp/assets/inputs/z.jpg', help='path to input image file')
    parser.add_argument('--out_dir', default='../autodl-tmp/assets/outputs', help='path to output image file')
    parser.add_argument('--out_name', default='z.jpg', help='output image name')
    parser.add_argument('--edit_txt', default='remove the zebra on the far right', help='prompts')
    parser.add_argument('--sam_ckpt', default='../autodl-tmp/sam_vit_h_4b8939.pth', help='path to origin SAM ckpt')
    parser.add_argument('--sam_type', default='vit_h', choices=['vit_l', 'vit_h', 'vit_b'], help='choose SAM model type')
    parser.add_argument('--seem_cfg', default='seem/configs/seem/focall_unicl_lang_demo.yaml', help='path to seem config file')
    parser.add_argument('--seem_ckpt', default='../autodl-tmp/seem_focall_v0.pt', help='path to origin SEEM ckpt')
    parser.add_argument('--inpaint_folder', default='../autodl-tmp/models/gqa_inpaint', help='path to origin gqa_inpaint ckpt folder')
    parser.add_argument('--inpaint_config', default='./paint/inpaint.yaml', help='path to amended gqa_inpaint config path')
    
    parser.add_argument('--example_config', default='./configs/v1.yaml', help='config path to Paint-by-Example')
    parser.add_argument('--example_ckpt', default='../autodl-tmp/model.ckpt', help='ckpt path to Paint-by-Example')
    
    # parser.add_argument('--sampler', default='DDIM', help='the method for sampling')
    # parser.add_argument('--sd_config', default='./config/', help='your stable diffusion model config')
    # parser.add_argument('--sd_ckpt', default='../autodl-tmp/sd-v1-5.ckpt', help='your stable diffusion model checkpoints')


    return parser.parse_args() 