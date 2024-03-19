from .guide import (
        get_bot, get_response, 
        planning_system_prompt, 
        planning_system_first_ask, 
        system_prompt_add_test, 
        system_prompt_remove_test
    )

import matplotlib.image as mpimg
from openai import OpenAI
import base64, requests
import pandas as pd
from .guide import 
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

def set_system(system_prompt, image_encoded):
    
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
    print(response.json())

if __name__ == "__main__":
    from time import time
    s = time()
    # path = os.listdir('../autodl-tmp/COCO/tset2017')
    # path = [os.path.join('../autodl-tmp/COCO/', '')]
    path = ['../assets/dog.jpg', '../assets/field.jpg']
    for p in path:
        p = encode_image(p)
        set_system(system_prompt_add_test, p)
    e = time()
    print(f'\n\ntime cost: {e - s}')

