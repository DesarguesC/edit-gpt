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
    # parser.add_argument('--out_name', default='z.jpg', help='output image name')
    parser.add_argument('--edit_txt', default='remove the zebra on the far right', help='prompts')
    parser.add_argument('--sam_ckpt', default='../autodl-tmp/sam_vit_h_4b8939.pth', help='path to origin SAM ckpt')
    parser.add_argument('--sam_type', default='vit_h', choices=['vit_l', 'vit_h', 'vit_b'], help='choose SAM model type')
    parser.add_argument('--seem_cfg', default='seem/configs/seem/focall_unicl_lang_demo.yaml', help='path to seem config file')
    parser.add_argument('--seem_ckpt', default='../autodl-tmp/seem_focall_v0.pt', help='path to origin SEEM ckpt')
    parser.add_argument('--inpaint_folder', default='../autodl-tmp/models/gqa_inpaint', help='path to origin gqa_inpaint ckpt folder')
    parser.add_argument('--inpaint_config', default='./paint/inpaint.yaml', help='path to amended gqa_inpaint config path')
    # paint-by-example
    parser.add_argument('--example_config', default='./configs/v1.yaml', help='config path to Paint-by-Example')
    parser.add_argument('--example_ckpt', default='../autodl-tmp/model.ckpt', help='ckpt path to Paint-by-Example')

    parser.add_argument('--depth_adapter_path', default='../autodl-tmp/t2iadapter/t2iadapter_depth_sd15v2.pth', help='ckpt path to depth adapter')
    parser.add_argument('--style_adapter_path', default='../autodl-tmp/t2iadapter/coadapter-style-sd15v1.pth', help='ckpt path to style adapter')
    # test different adapter type ? seg / depth / keypose / openpose is ok. To fit sd1.5, we use depth first
    
    parser.add_argument('--lama_config', default='./configs/lama_default.yaml', help='path to lama inpainting config path')
    parser.add_argument('--lama_ckpt', default='../autodl-tmp/lama/', help='path to lama ckpt folder')
    parser.add_argument('--XL_base_path', default='../autodl-tmp', help='base path to XL, adapter, etc.')

    parser.add_argument('--example_type', type=str, default='XL', help="choose the method for generation",
        choices=['XL', 'XL_adapter', 'v1.5', 'v1.5_adapter'],
    )
    parser.add_argument('--expand_scale', default=1.0, type=float, help='expansion scale for mask')


    return parser.parse_args() 