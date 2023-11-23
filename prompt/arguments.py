import argparse


def get_args():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", help='path to config file', )
    # set as default
    parser.add_argument('--in_dir', default='../autodl-tmp/assets/inputs/z.jpg', help='path to input image file')
    parser.add_argument('--out_dir', default='../autodl-tmp/assets/outputs', help='path to output image file')
    parser.add_argument('--name', default='z.jpg', help='output image name')
    parser.add_argument('--edit_txt', default='remove the zebra on the far right', help='prompts')
    parser.add_argument('--sam_ckpt', default='../autodl-tmp/sam_vit_h_4b8939.pth', help='path to origin SAM ckpt')
    parser.add_argument('--sam_type', default='vit-h', choises=['vit-l', 'vit-h', 'vit-b'], help='choose SAM model type')
    parser.add_argument('--seem_cfg', default='seem/configs/seem/focall_unicl_lang_demo.yaml', help='path to seem config file')
    parser.add_argument('--seem_ckpt', default='../autodl-tmp/seem_focall_v0.pt', help='path to origin SEEM ckpt')
    

    return parser.parse_args() 