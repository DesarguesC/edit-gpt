from guide import (
        get_bot, get_response,
        system_prompt_add_test, 
        system_prompt_remove_test,
        task_planning_test_system_prompt
    )
# .guide or prompt.guide ?

from openai import OpenAI
import base64, requests, csv
import pandas as pd

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

TYPE = {'add', 'replace', 'remove', 'move'}

api_key = list(pd.read_csv('./key.csv')['key'])[0]
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

def csv_writer(csv_path, one_dict):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        # Write header
        writer.writeheader()
        # Write data
        if not isinstance(one_dict, list):
            writer.writerow(one_dict)
        else:
            for dict_ in one_dict:
                writer.writerow(one_dict)

import os
from time import time
from random import randint
from jieba import re
from PIL import Image
if __name__ == "__main__":

    tot_image_num = 1
    prompts_num_per_image = 10

    s = time()
    path_base = '../autodl-tmp/COCO/val2017'
    img_copy_path = '../autodl-tmp/GPT/GPT_img'
    output_path = '../autodl-tmp/GPT/GPT_gen_raw'
    labeled_path = '../autodl-tmp/GPT/GPT_gen_label' # not used in this section
    path_list = os.listdir(path_base)
    length = len(path_list)
    selected_list = []
    while len(selected_list) < tot_image_num:
        idx = randint(0, length)
        while idx in selected_list: idx = randint(0, length)
        selected_list.append(idx)
    string_dict = {}
    for i in range(len(selected_list)):
        idx = selected_list[i]
        img_path = os.path.join(path_base, f'{idx:0{12}}.jpg')
        img_encoded = encode_image(img_path)
        string = []
        # TODO: write into csv
        # for i in range(prompts_num_per_image):
            
        data = gpt4v_response(task_planning_test_system_prompt, img_encoded,json_mode=False)
        description = [ prompt for prompt in re.split(r"[(),]", data.choices[0].message.content.strip()) is prompt not in ['',' ']]
        print(f'len of description: {len(description)}')
        for i in range(len(description)):
            print(f'\t{i}: {description[i]}')
            string_dict[str(i)] = description[i]
        
        dict_list = pd.DataFrame(string_dict).to_dict(orient='records')
        assert(len(dict_list))
        csv_writer(os.path.join(output_path, f'{i}:0{3}.csv'), dict_list)
        (Image.open(img_path).convert('RGB')).save(os.path.join(img_copy_path, f'{i:0{3}}.jpg'))


    e = time()
    print(f'\n\ntime cost: {e - s}')
    print('Done.')

    """
    Generated Raw Data
        GPT_gen_raw
            |
            |---000.csv (10 prompts)
            |---001.csv (10 prompts)
            |--- ... 
        
    Labeled Data
        GPT_gen_labeled
            |
            |---000.csv (10 columns)
            |---001.csv (10 columns)
            |--- ...
            
            0, 1, 2, ..., 9
            add, remove, replace, ...
            ..., ..., ..., ...
            [a column is a set of plans of a complex prompt]
    """

