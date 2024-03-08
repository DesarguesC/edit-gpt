import numpy as np
from PIL import Image, ImageOps
from omegaconf import OmegaConf
from pathlib import Path
import os,sys, torch, yaml, glob, argparse
from paint.crutils import ab8, ab64

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

from paint.utils import load_img_to_array, save_array_to_img

def get_reshaped_img(opt, img_pil=None):

    img_pil = Image.open(opt.in_dir).convert('RGB') if img_pil is None else img_pil
    w, h = img_pil.size
    if not hasattr(opt, 'W'): setattr(opt, 'W', ab64(w))
    else: opt.W = ab64(w)
    if not hasattr(opt, 'H'): setattr(opt, 'H', ab64(h))
    else: opt.H = ab64(h)
    if opt.W != w and opt.H != h:
        img_pil = ImageOps.fit(img_pil, (w, h), method=Image.Resampling.LANCZOS)
    
    return opt, img_pil


@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod = 8,
        device = "cuda",
        preloaded_lama_remover = None
    ):
    if isinstance(mask, list):
        print(f'len(mask) = {len(mask)}')
        for i in range(len(mask)):
            print(f'mask[{i}].shape = {mask[i].shape}')
    else: print(f'mask.shape = {mask.shape}')
    
    # for mask_ in mask:
        # assert len(mask_.shape) == 2
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    if np.max(mask) == 1:
        mask = mask * 255
    device = torch.device(device)

    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    print(' '*6+'-'*9+'loading lama'+'-'*9)
    
    if preloaded_lama_remover is None:

        predict_config = OmegaConf.load(config_p)
        predict_config.model.path = ckpt_p
        
        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

    else:
        model = preloaded_lama_remover['model']
        predict_config = preloaded_lama_remover['predict_config']

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model


@torch.no_grad()
def inpaint_img_with_builded_lama(
        model,
        img: np.ndarray,
        mask: np.ndarray,
        config_p=None,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

