from prompt.guide import (
        get_bot, get_response,
        system_prompt_add_test, 
        system_prompt_remove_test,
        task_planning_test_system_prompt
    )

from prompt.gpt4_gen import gpt4v_response
import csv, os, json
import pandas as pd
from time import time
from random import randint
from jieba import re
from PIL import Image


def csv_writer(csv_path, one_dict):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=one_dict.keys())
        # Write header
        writer.writeheader()
        # Write data
        if not isinstance(one_dict, list):
            writer.writerow(one_dict)
        else:
            for dict_ in one_dict:
                writer.writerow(one_dict)


if __name__ == "__main__":


    tot_num = 500
    api_key = list(pd.read_csv('./key.csv')['key'])[0]
    net_proxy = 'http://127.0.0.1:7890'
    engine = "gpt-4"
    # engine = 'gpt-3.5-turbo'
    agent = get_bot(engine=engine, api_key=api_key, system_prompt=task_planning_test_system_prompt, proxy=net_proxy)

    s = time()
    path_base = '../autodl-tmp/COCO/val2017'
    img_copy_path = '../autodl-tmp/GPT/GPT_img'
    output_path = '../autodl-tmp/GPT/GPT_gen_raw'
    labeled_path = '../autodl-tmp/GPT/GPT_gen_label' # not used in this section

    for paths in [img_copy_path, output_path, labeled_path]:
        if os.path.exists(paths): os.system(f'rm {paths}/*')
        else: os.mkdir(paths)

    # Image Caption Hash Map
    with open('../autodl-tmp/COCO/annotations/captions_val2017.json') as f:
        captions = json.load(f)

    captions_dict = {}
    for x in captions['annotations']:
        image_id = str(x['image_id'])
        if image_id in captions_dict:
            captions_dict[image_id] = captions_dict[image_id].strip() + '; ' + x['caption'].strip()
        else:
            captions_dict[image_id] = x['caption']
    
    selected_list = []
    key_list = list(captions_dict.keys())
    length = len(key_list)
    while len(selected_list) < tot_num:
        idx = randint(0, length)
        while idx in selected_list:
            idx = randint(0, length)
        selected_list.append(idx)

    for i in range(len(selected_list)):
        try:
            idx = selected_list[i]
            string_dict = {}
            caption_input = captions_dict[key_list[idx]]
            print(f'\n\n{caption_input}\n\n')
            data = get_response(agent, caption_input)
            print(f'\n\nraw data: \n{data}\n\n')
            description = [ prompt for prompt in re.split(r"[\(\)|.]", data.strip()) if prompt not in ['',' ', '\n']]
            # print(f'Number of Prompts from GPT: {len(description)}')
            for j in range(len(description)):
                print(f'\t{j}: {description[j]}')
                string_dict[str(j)] = description[j]
            
            # dict_list = pd.DataFrame(string_dict).to_dict()
            # print(dict_list)
            csv_path = os.path.join(output_path, f'{i:0{5}}.csv')
            if os.path.isfile(csv_path): os.system(f'rm {csv_path}')
            csv_writer(csv_path, string_dict)
            Image.open(os.path.join(path_base, f'{int(key_list[idx]):0{12}}.jpg')).save(os.path.join(img_copy_path, f'{i:0{5}}.jpg'))
        except Exception as err:
            string = f'Error Occurred: {err}'
            print(string)

    e = time()
    print(f'\n\ntime cost: {e - s}')
    print('Done.')

    """
    Generated Raw Data
        GPT_gen_raw
            |
            |---000.csv (prompts series)
            |---001.csv (prompts series)
            |--- ... 
        
    Labeled Data
        GPT_gen_labeled
            |
            |---000.csv 
            |---001.csv 
            |--- ...
            
            0, 1, 2, ...,
            add, remove, replace, ...
            ..., ..., ..., 
            [a column is a method arranged]
    """


