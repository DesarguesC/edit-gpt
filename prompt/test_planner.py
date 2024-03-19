from .guide import (
        get_bot,
        get_response,
        planning_system_prompt,
        planning_system_first_ask,
    )
import pandas as pd
from jieba import re
import os
from matplotlib import pyplot as plt

labeled_path = '../autodl/GPT_gen_labeled'

def get_planner():
    engine = 'gpt-3.5-turbo'
    api_key = list(pd.read_csv('./key.csv'))[0]
    proxy = 'http://127.0.0.1:7890'
    agent = get_bot(engine=engine, api_key=api_key, system_prompt=planning_system_prompt, proxy=proxy)
    _ = get_response(agent, planning_system_first_ask)
    return agent

def evaluate(agent, prompt_csv_path, label_csv_path):
    """
        raw csv ~ label csv
        calculate targets: average accuracy, accuracy (average, for a length) ~ answer series length
    """
    input_df = list(pd.read_csv(prompt_csv_path))
    ans_df = pd.read_csv(label_csv_path)
    acc_list_tot = []

    for i in range(10):
        raw_plan = get_response(agent, list(input_df[str(i)])[0])
        raw_plan = [x.strip() for x in re.spilt(r"[;]", raw_plan) if x != '' and x != ' ']
        for j in range(len(raw_plan)):
            raw_plan[j] = [x.strip() for x in re.spilt(r"[,]", raw_plan[j]) if x != '' and x != ' '][0].lower()
        ans_list = list(ans_df[str(i)])
        acc_list = []
        for i in range(0, min(len(ans_list, len(raw_plan)))):
            acc_list.append( int(ans_list[i].lower().strip() == raw_plan[i]) / (i+1) )
        for i in range(min(len(ans_list, len(raw_plan))), max(len(ans_list, len(raw_plan)))):
            acc_list.append(0.)

        acc_list_tot.append(acc_list)
    # 每张图片的，每条prompt的准确率，以及准确率随操作总数的关系
    return acc_list_tot

if __name__ == "__main__":
    agent = get_planner()
    raw_path = '../autodl-tmp/GPT_gen_raw'
    label_path = '../autodl-tmp/GPT_gen_label'
    raw_file_list, label_file_list = os.listdir(raw_path), os.listdir(label_path)
    assert len(raw_file_list) == len(label_file_list), f'len(raw_file_list) = {len(raw_file_list)}, len(label_file_list) = {len(label_file_list)}'
    img_acc_list = []
    for i in range(len(raw_file_list)):
        raw_csv_path = os.path.join(raw_path, raw_file_list[i])
        label_csv_path = os.path.join(label_path, label_file_list[i])
        img_acc_list.append(evaluate(agent, raw_csv_path, label_csv_path))

    # TODO: save the result, draw the fiture

