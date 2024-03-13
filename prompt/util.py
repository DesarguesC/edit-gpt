import torch, cv2
from PIL import Image

def get_image_from_box(image, box):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(image, str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(box)==4, f'box = {box}'
    # print(f'type-box = {type(box)}')
    assert len(image.shape)==3, f'image.shape = {image.shape}'
    x, y, w, h = box[0], box[1], box[2], box[3]
    box_image = image[y:y+h,x:x+w,:]
    # box: x,y,w,h   |   image: h,w,c
    # print(f'box_image.shape = {box_image.shape}')
    return box_image

import torch, os
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from functools import partial
from datasets import load_dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from zipfile import ZipFile
from PIL import Image
from basicsr.utils import tensor2img, img2tensor


"""
    FID results tend to be fragile as they depend on a lot of factors:
        The specific Inception model used during computation.
        The implementation accuracy of the computation.
        The image format (not the same if we start from PNGs vs JPGs).
    Keeping that in mind, FID is often most useful when comparing similar runs, 
    but it is hard to reproduce paper results unless the authors carefully disclose the FID measurement code.
    These points apply to other related metrics too, such as KID and IS.
"""

def Cal_FIDScore(real_image_list, fake_image_list):
    # real_image: list(np.array), with the same shape (512,512) or (256,256)
    # fake_image: list(np.array), with the same shape (512,512) or (256,256)
    assert isinstance(fake_image_list, list) and isinstance(real_image_list, list)

    real_images = torch.cat([tensor2img(img) for img in real_image_list], dim=0).detach().cpu().numpy()
    fake_images = torch.cat([tensor2img(img) for img in fake_image_list], dim=0).detach().cpu().numpy()

    fid = FID(normalize=True)
    fid.update(real_images, real=True)
    fid_inited.update(fake_images, real=False)

    return float(fid.compute())

import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

class DirectionalSimilarity(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device
        clip_id = '../autodl-tmp/openai/clip-vit-large-patch14'
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_id)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(self.device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(self.device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity


def Cal_ClipDirectionalSimilarity(image_before_list, image_after_list, caption_before_list, caption_after_list):
    dir_similarity = DirectionalSimilarity()
    scores = []
    assert len(image_before_list) == len(image_after_list) and len(caption_before_list) == len(caption_after_list) and len(image_after_list) == len(caption_after_list), \
            f'len(image_before_list) = {len(image_before_list)}, len(image_after_list) = {len(image_after_list)}, '\
            f'len(caption_before_list) = {len(caption_before_list)}, len(caption_after_list) = {len(caption_after_list)}'
    for i in range(len(image_before_list)):
        scores.append(dir_similarity(
            image_before_list[i], image_after_list[i], 
            caption_before_list[i], caption_after_list[i]
        ).detach().cpu().numpy())

    return np.mean(np.array(scores))


