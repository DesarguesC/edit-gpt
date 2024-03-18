from guide import get_bot, get_response
import matplotlib.image as mpimg
from openai import OpenAI
import base64, requests
import pandas as pd

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

TYPE = {'add', 'replace', 'remove', 'move'}

"""
    你是一个图片感知机，我需要你针对输入的图片做两件事情：1. 生成一条图片编辑指令； 2. 根据你设置的图片编辑指令设置一条询问这个编辑是否完成的一般疑问句。
    生成编辑指令时，编辑对象必须是图片中存在的物体，且指令必须在移动物体对象且带有方位信息，例如：“将苹果移动到桌子下面”，但前提是图片里有桌子。
    在生成疑问句时候，例如前面你生成的编辑指令是“将苹果移动到桌子下面”，那么生成的疑问句应当是：“苹果是不是在桌子下面？”，注意不要带有动词，尽量使用介词。
    请不要使用过度复杂的方位词信息，我们认为“将苹果移动到桌子下面”已经是一个足够复杂的句子了（请生成复杂程度相当的指令）。
    另外在生成时，编辑指令和疑问句各占一行，一共两行，禁止出现多余的字符。

"""

system_prompt_add_test = "You are a picture-aware machine, and I need you to do two things with an input picture: "\
                         "1. generate a picture editing instruction, and 2. set a general question asking if this edit is complete, "\
                         "based on the picture editing instruction you set. \nWhen generating an edit command, "\
                         "the object to be edited must be an object that exists in the picture, "\
                         "and the command must move the object with orientation information, for example: \"Move the apple under the table\""\
                         ", but only if there is a table in the picture. \nWhen generating a question, "\
                         "for example, if you generated an editorial instruction \"Move apples under the table\", "\
                         "then the question should be \"Are apples under the table?\". Be careful not to use verbs, "\
                         "and try to use prepositions. Please do not use overly complex orientation information. "\
                         "For instance, we think \"move the apple under the table\" is a complex enough sentence "\
                         "(please generate instructions of comparable complexity).\nNote that, when generating, the editorial instruction and "\
                         "the interrogative sentence each take up one line, totaling two lines, and superfluous characters are prohibited."

system_prompt_remove_test = "You are a picture-aware machine, and I need you to do two things with an input picture: "\
                         "1. generate a picture editing instruction, and 2. set a general question asking if this edit is complete, "\
                         "based on the picture editing instruction you set. \nWhen generating an edit command, "\
                         "the object to be edited must be an object that exists in the picture, "\
                         "and the command must move the object with orientation information, for example: \"Move the apple under the table\""\
                         ", but only if there is a table in the picture. \nWhen generating a question, "\
                         "for example, if you generated an editorial instruction \"Move apples under the table\", "\
                         "then the question should be \"Are apples under the table?\". Be careful not to use verbs, "\
                         "and try to use prepositions. Please do not use overly complex orientation information. "\
                         "For instance, we think \"move the apple under the table\" is a complex enough sentence "\
                         "(please generate instructions of comparable complexity).\nNote that, when generating, the editorial instruction and "\
                         "the interrogative sentence each take up one line, totaling two lines, and superfluous characters are prohibited."

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

