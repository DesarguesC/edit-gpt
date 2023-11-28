from typing import Tuple
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import os
from omegaconf import OmegaConf
import importlib

to_tensor = ToTensor()

inpaint_config_path = './configs/latent-diffusion/gqa-inpaint-ldm-vq-f8-256x256.yaml'
inpaint_model_base_path = './inst-paint'



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_inpaint_model(ckpt_base_path=inpaint_model_base_path, config_path=inpaint_config_path, device='cuda'):
    parsed_config = OmegaConf.load(config_path)
    model = instantiate_from_config(parsed_config["model"])
    model_state_dict = torch.load(os.path.join(ckpt_base_path, 'ldm/model.ckpt'), map_location="cuda")["state_dict"]
    print(os.path.join(ckpt_base_path, 'ldm/model.ckpt'))
    print(os.path.isfile(os.path.join(ckpt_base_path, 'ldm/model.ckpt')))
    print('ldm model.ckpt loaded')
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)


def preprocess_image(
        image: Image, resize_shape: Tuple[int, int] = (256, 256), center_crop=False
):
    pil_image = image

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        pil_image = image.crop((left, top, right, bottom))

    pil_image = pil_image.resize(resize_shape)

    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0) * 2 - 1

    return pil_image, tensor_image


def target_removing(
        opt, target_noun: str, image: Image, model=None, resize_shape: Tuple[int, int] = (256, 256),
        ori_shape: Tuple[int, int] = (512, 512), recovery=True, center_crop=False
) -> Image:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ori_shape => image.shape
    model = load_inpaint_model(ckpt_base_path=opt.inpaint_folder, config_path=opt.inpaint_config, device=device) if model==None else model
    print(f'model: \n{model}')
    pil_image_pointer = image

    if center_crop:
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2
        pil_image_pointer = image.crop((left, top, right, bottom))
        # if used, connect it with the uncopped part

    pil_image = pil_image_pointer.resize(resize_shape)
    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0) * 2 - 1
    print(f'tensor_image.shape = {tensor_image.shape}')
    rmtxt = 'remove the ' + target_noun
    pil_removed = model.inpaint(tensor_image, rmtxt, num_steps=50, device=device, return_pil=True, seed=0)
    if recovery: pil_removed = pil_removed.resize(ori_shape)
    return pil_removed





