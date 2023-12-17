import cv2, torch
import numpy as np
from typing import Tuple


def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes an image and its corresponding mask to have the longer side equal to `target_size` and pads them to make them
    both have the same size. The resulting image and mask have dimensions (target_size, target_size).

    Args:
        image: A numpy array representing the image to resize and pad.
        mask: A numpy array representing the mask to resize and pad.
        target_size: An integer specifying the desired size of the longer side after resizing.

    Returns:
        A tuple containing two numpy arrays - the resized and padded image and the resized and padded mask.
    """
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    mask_padded = np.pad(mask_resized, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    return image_padded, mask_padded, (top_pad, bottom_pad, left_pad, right_pad)

def recover_size(image_padded: np.ndarray, mask_padded: np.ndarray, orig_size: Tuple[int, int], 
                 padding_factors: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes a padded and resized image and mask to the original size.

    Args:
        image_padded: A numpy array representing the padded and resized image.
        mask_padded: A numpy array representing the padded and resized mask.
        orig_size: A tuple containing two integers - the original height and width of the image before resizing and padding.

    Returns:
        A tuple containing two numpy arrays - the recovered image and the recovered mask with dimensions `orig_size`.
    """
    h,w,c = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors
    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    mask = mask_padded[top_pad:h-bottom_pad, left_pad:w-right_pad]
    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    return image_resized, mask_resized


import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)

def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point


def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        step: int = 50,
        device="cuda"
):
    masks.astype(np.uint8) * 255
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    img_padded = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(255 - mask_padded),
        num_inference_steps=step,
    ).images[0]
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1-mask_resized) + img * mask_resized
    return img_resized



@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod=8,
        device="cuda"
):
    # assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    masks.astype(np.uint8)    
    
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
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
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

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

