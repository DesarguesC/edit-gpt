from prompt.guide import (
        get_bot, get_response,
        system_prompt_add_test, 
        system_prompt_remove_test,
        task_planning_test_system_prompt
    )
# .guide or prompt.guide ?

from openai import OpenAI
import base64, requests, csv, os
import pandas as pd

from time import time
from random import randint
from jieba import re
from PIL import Image

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

def gpt4v_response(system_prompt='', edit_prompt=None, image_encoded=None, has_encoded=True, json_mode=True):
    if not has_encoded:
        assert isinstance(image_encoded, str)
        image_encoded = encode_image(image_encoded)
    
    content = [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_encoded}",
                            "detail": "high"
                        }
                    }]
    if edit_prompt is not None:
        content.append({
                        "type": "text",
                        "text": edit_prompt
                    })

    payload = {
        "model": 'gpt-4-vision-preview',
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content":content
            }
    ],
        "max_tokens": 600
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, proxies=proxy_dict)
    return response.json() if json_mode else response




gpt4_v_get_box = "You are a bounding box generator. I'm giving you a image and a editing prompt. The prompt is to move a target object to another place, "\
                 "such as \"Move the apple under the desk\", \"move the desk to the left\". "\
                 "What you should do is to return a proper bounding box for it. The output should be in the form of $[Name, (X,Y,W,H)]$"\
                 "For instance, you can output $[\"apple\", (200, 300, 20, 30)]$. Your output cannot contain $(0,0,0,0) as bounding box. $"

def gpt_4v_bbox_return(image_path, edit_prompt):
    image_encoded = encode_image(image_path)
    response = gpt4v_response(gpt4_v_get_box, edit_prompt, image_encoded, json_mode=True)
    print('response: \n', response)
    return response['choices'][0]['message']['content']


