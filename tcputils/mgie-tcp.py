import os, time
from tqdm.auto import tqdm
from PIL import Image
import cv2
import torch as T
import transformers, diffusers
from llava.conversation import conv_templates
from llava.model import *
from socket import *
import threading
import numpy as np

def crop_resize(f, sz=512):
    w, h = f.size
    if w>h:
        p = (w-h)//2
        f = f.crop([p, 0, p+h, h])
    elif h>w:
        p = (h-w)//2
        f = f.crop([0, p, w, p+w])
    f = f.resize([sz, sz])
    return f
def remove_alter(s):  # hack expressive instruction
    if 'ASSISTANT:' in s: s = s[s.index('ASSISTANT:')+10:].strip()
    if '</s>' in s: s = s[:s.index('</s>')].strip()
    if 'alternative' in s.lower(): s = s[:s.lower().index('alternative')]
    if '[IMG0]' in s: s = s[:s.index('[IMG0]')]
    s = '.'.join([s.strip() for s in s.split('.')[:2]])
    if s[-1]!='.': s += '.'
    return s.strip()



def main():
    DEFAULT_IMAGE_TOKEN = '<image>'
    DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
    DEFAULT_IM_START_TOKEN = '<im_start>'
    DEFAULT_IM_END_TOKEN = '<im_end>'
    PATH_LLAVA = '../autodl-tmp/LLaVA-7B-v1'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
    model = LlavaLlamaForCausalLM.from_pretrained(PATH_LLAVA, low_cpu_mem_usage=True, torch_dtype=T.float16, use_cache=True).cuda()
    image_processor = transformers.CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=T.float16)
    
    tokenizer.padding_side = 'left'
    tokenizer.add_tokens(['[IMG0]', '[IMG1]', '[IMG2]', '[IMG3]', '[IMG4]', '[IMG5]', '[IMG6]', '[IMG7]'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    ckpt = T.load('./_ckpt/mgie_7b/mllm.pt', map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    
    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end: tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    vision_tower = model.get_model().vision_tower[0]
    vision_tower = transformers.CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=T.float16, low_cpu_mem_usage=True).cuda()
    model.get_model().vision_tower[0] = vision_tower
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end: vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size//vision_config.patch_size)**2
    
    _ = model.eval()
    EMB = ckpt['emb'].cuda()
    with T.inference_mode(): NULL = model.edit_head(T.zeros(1, 8, 4096).half().to('cuda'), EMB)
    print('NULL:', NULL.shape)

    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained('../autodl-tmp/ip2p', torch_dtype=T.float16, safety_checker=None).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.load_state_dict(T.load('./_ckpt/mgie_7b/unet.pt', map_location='cpu'))

    
    # TCP Sender Utils
    def receive_from_length(socks, length):
        buf = b''
        while length > 0:
            new_buf = socks.recv(length)
            if not new_buf:
                return None
            buf = buf + new_buf
            length = length - len(new_buf)
        return buf
        
    def receive_image_from_length(socks, length=16):
        length = receive_from_length(socks, length)
        img_str = receive_from_length(socks, int(length))
        data = np.fromstring(img_str, dtype='uint8')
        decode_img = cv2.imdecode(data, 1)
        return Image.fromarray(decode_img).convert('RGB')
        
    def Encode_and_Send(socks, img_Image):
        image = np.array(img_Image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        result, img_encode = cv2.imencode('.jpg', image, encode_param)
        data = np.array(img_encode)
        str_data = data.tostring()
        socks.send(str(len(str_data)).ljust(16).encode())
        print('Length Sent.')
        socks.send(str_data)
        print('Image Sent.')
    
        
    
    # MGIE Utils
    def input_preprocess(edit_str, img_Image):
        img_Tensor = image_processor.preprocess(img_Image, return_tensors='pt')['pixel_values'][0]
        edit_str = f'what will this image be like if {edit_str}\n'\
                    f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_PATCH_TOKEN*image_token_len}{DEFAULT_IM_END_TOKEN}'
        
        conv = conv_templates['vicuna_v1_1'].copy()
        conv.append_message(conv.roles[0], edit_str), conv.append_message(conv.roles[1], None)

        edit_str = tokenizer(conv.get_prompt())
        edit_str, edit_mask = T.as_tensor(edit_str['input_ids']), T.as_tensor(edit_str['attention_mask'])
        return edit_str, img_Tensor, edit_mask

        
        

    SEED = 42
    serverHost, serverPort = '127.0.0.1', 4096
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((serverHost, serverPort))
    print(f'\n\nMGIE Socket is now available as {serverHost}:{serverPort}\n\n')
    serverSocket.listen(1)
    connectionSocket, addr = serverSocket.accept()


    while True:
        # edit_txt = sock.recv(1024).decode('utf-8') # str
        edit_txt = connectionSocket.recv(4096).decode()
        print(f'Edit_txt received: {edit_txt}')
        time.sleep(0.5)
        img = receive_image_from_length(connectionSocket) # Image.Image
        print(f'Received: img.size = {img.size}')

        edit_txt, edit_img, edit_mask = input_preprocess(edit_txt, img)
        with T.inference_mode():
            out = model.generate(
                        edit_txt.unsqueeze(dim=0).cuda(),
                        images = edit_img.half().unsqueeze(dim=0).cuda(),
                        attention_mask = edit_mask.unsqueeze(dim=0).cuda(),
                        do_sample = False, 
                        max_new_tokens = 96, 
                        num_beams = 1, 
                        no_repeat_ngram_size = 3, 
                        return_dict_in_generate = True, 
                        output_hidden_states = True
                    )
            out, hid = out['sequences'][0].tolist(), T.cat([x[-1] for x in out['hidden_states']], dim=1)[0]
            p = min(out.index(32003)-1 if 32003 in out else len(hid)-9, len(hid)-9)
            hid = hid[p:p+8]
            
            out = remove_alter(tokenizer.decode(out))
            emb = model.edit_head(hid.unsqueeze(dim=0), EMB)
            res = pipe(image=img, prompt_embeds=emb, negative_prompt_embeds=NULL, generator=T.Generator(device='cuda').manual_seed(SEED)).images[0] # Image.Image

        Encode_and_Send(connectionSocket, res)
        time.sleep(2)
        

    serverSocket.close()
    sys.exit()
    
    

        
if __name__ == '__main__':
    main()