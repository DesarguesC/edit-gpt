from .guide import (
        get_bot, get_response, 
        planning_system_prompt, 
        planning_system_first_ask, 
        system_prompt_add_test, 
        system_prompt_remove_test,
        task_planning_test_system_prompt
    )
# .guide or prompt.guide ?

from openai import OpenAI
import base64, requests
import pandas as pd

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

TYPE = {'add', 'replace', 'remove', 'move'}

api_key = list(pd.read_csv('../key.csv')['key'])[0]
headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
proxy_dict = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

def gpt4v_response(system_prompt, image_encoded, json_mode=True):
    
    payload = {
        "model": 'gpt-4-vision-preview',
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content":[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_encoded}",
                            "detail": "high"
                        }
                    }]
            }
    ],
        "max_tokens": 300
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, proxies=proxy_dict)
    return response.json() if json_mode else response



import os
from time import time
from random import randint
if __name__ == "__main__":

    tot_image_num = 50
    prompts_per_image = 10

    s = time()
    path_base = '../autodl-tmp/COCO/val2017'
    output_path = '../autodl-tmp/GPT_gen_raw'
    # labeled path = '../autodl-tmp/GPT_gen_label'
    path_list = os.listdir(path_base)
    length = len(path_list)
    selected_list = []
    while len(selected_list) < tot_image_num:
        idx = randint(0, length)
        while idx in selected_list: idx = randint(0, length)

    for idx in selected_list:
        img_path = os.path.join(path_base, f'{idx}:0{12}.jpg')
        img_encoded = encode_image(img_path)
        string = ''
        for i in range(prompts_per_image):
            data = gpt4v_response(task_planning_test_system_prompt, img_encoded,json_mode=False)
            description = data.choices[0].message.content
            string = string + description + '\n'
        with open(os.path.join(output_path, f'{idx}:0{3}.txt'), 'w') as f:
            f.write(string)

    e = time()
    print(f'\n\ntime cost: {e - s}')
    print('Done.')

