import torch, argparse, cv2
import numpy as np
from basicsr.utils import img2tensor
from ldm.modules.encoders.adapter import Adapter, StyleAdapter, Adapter_light
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict, resize_numpy_image
from PIL import Image

def get_adapter(opt, adapter_type='depth'):
    adapter = {}
    adapter['cond_weight'] = opt.cond_weight # cond_weight = 1.0

    adapter['model'] = Adapter(
        cin=64 * 3, # sketch / canny: 64 * 1
        channels=[320, 640, 1280, 1280][:4],
        nums_rb=2,
        ksize=1,
        sk=True,
        use_conv=False).to(opt.device)

    adapter['model'].load_state_dict(torch.load(opt.depth_adapter_path if adapter_type=='depth' else opt.style_adapter_path))
    return adapter




def get_depth_model(opt):
    from ldm.modules.extra_condition.midas.api import MiDaSInference
    model = MiDaSInference(model_type='dpt_hybrid').to(opt.device)  # get cond model
    return model

def process_depth_cond(opt, cond_image: Image = None, cond_model=None) -> torch.Tensor:
    if cond_model is None:
        return cond_iamge
    if (opt.W, opt.H) != cond_image.size:
        cond_image = cond_image.resize((opt.W, opt.H))
    depth = cv2.cvtColor(np.array(cond_image), cv2.COLOR_RGB2BGR)
    # depth = resize_numpy_image(depth, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    # opt.H, opt.W = depth.shape[:2]
    depth = img2tensor(depth).unsqueeze(0) / 127.5 - 1.0
    depth = cond_model(depth.to(opt.device)).repeat(1, 3, 1, 1)
    depth -= torch.min(depth)
    depth /= torch.max(depth)
    return depth

def get_style_model(opt):
    from transformers import CLIPProcessor, CLIPVisionModel
    version = 'openai/clip-vit-large-patch14'
    processor = CLIPProcessor.from_pretrained(version)
    clip_vision_model = CLIPVisionModel.from_pretrained(version).to(opt.device)
    return {'processor': processor, 'clip_vision_model': clip_vision_model}



def process_style_cond(opt, cond_image: Image = None, cond_model=None) -> torch.Tensor:
    style = Image.fromarray(cond_image)
    style_for_clip = cond_model['processor'](images=style, return_tensors="pt")['pixel_values']
    style_feat = cond_model['clip_vision_model'](style_for_clip.to(opt.device))['last_hidden_state']

    return style_feat


def get_adapter_feature(inputs, adapters):
    ret_feat_map = None
    ret_feat_seq = None
    if not isinstance(inputs, list):
        inputs = [inputs]
        adapters = [adapters]

    for input, adapter in zip(inputs, adapters):
        cur_feature = adapter['model'](input)
        if isinstance(cur_feature, list):
            if ret_feat_map is None:
                ret_feat_map = list(map(lambda x: x * adapter['cond_weight'], cur_feature))
            else:
                ret_feat_map = list(map(lambda x, y: x + y * adapter['cond_weight'], ret_feat_map, cur_feature))
        else:
            if ret_feat_seq is None:
                ret_feat_seq = cur_feature * adapter['cond_weight']
            else:
                ret_feat_seq = torch.cat([ret_feat_seq, cur_feature * adapter['cond_weight']], dim=1)

    return ret_feat_map, ret_feat_seq