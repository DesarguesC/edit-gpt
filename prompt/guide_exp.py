import os, time
from guide import get_response, get_bot


"""
    你是一个instruction生成器，你需要根据描述两幅相似图像的caption中的文字差异，生成一条能够通过“replace” 实现图像编辑的指令，例如：
    Input: 1.mountains of scotland under a bright sunny sky 2. mountains of scotland under a rainy sky
    Output: replace "bright sunny sky" with a "rainy sky".
    我们更希望你使用"replace A with B"的句型。另外，如果你认为这两条caption之间不能用一条只用“replace”方法的instruction实现，
    请输出“NULL”。注意，你的输出中禁止包含其他多余的字符。
"""

system_prompt_gen_replace_instructions = "You are an instruction generator, and you need to generate an instruction that "\
                                         "enables image editing via \"replace\" based on the textual differences in a "\
                                         "caption describing two similar images, for example: \n"\
                                         "Input: 1.mountains of scotland under a bright sunny sky 2. mountains of scotland under a rainy sky\n"\
                                         "Output: replace \"bright sunny sky\" with a \"rainy sky\". \n"\
                                         "We prefer you to use the \"replace A with B\" pattern. Also, if you think that the two captions "\
                                         "can't be separated by an instruction that only uses the \"replace\" method, just output \"NULL\". "\
                                         "Note that you are forbidden to include any other extra characters in your output."


# 输入：cpation, label, bounding box
"""
    你是一个位置生成器，你需要根据输入的caption和label，为处在bounding box描述位置处的物体生成一个使用文字描述的位置。
    你获得的输入是：caption，label，（x,y,w,h）
    这里的(x,y,w,h)是bounding box，其含义为：(x,y)表示bounding box左上角的点的坐标，(w,h)为bounding box的宽度和高度。
    你需要将生成的描述性位置输出，同时给出另一个目标位置的文字描述，使得物体可以从当前位置被移动到目标位置，例如：
    Input: an apple is on the desk, apple, (100,100,50,70)
    Output: on the desk; under the desk
    Input: an apple on the desk, desk, (30,140,300,240)
    Output: desk on the left; desk on the right
    每个输出中的两个位置A和B用";"分割，你的输出禁止包含多余的无关字符
"""


system_prompt_gen_move_instructions = "You are a position generator and you need to generate a textual position "\
                                      "for an object at the position described by a bounding box, based on the input caption and label. "\
                                      "The input you get is: caption, label, (x,y,w,h). Here (x,y,w,h) is the bounding box, "\
                                      "which means: (x,y) represents the coordinates of the point in the upper left corner of "\
                                      "the bounding box, and (w,h) is the width and height of the bounding box. "\
                                      "You need to output the generated descriptive position with another textual description "\
                                      "of the target position, so that the object can be moved from its current position to the target position, for example: \n"
                                      "Input: an apple is on the desk, apple, (100,100,50,70)\n"\
                                      "Output: on the desk; under the desk\n"\
                                      "Input: an apple on the desk, desk, (30,140,300,240)\n"\
                                      "Output: desk on the left; desk on the right\n"\
                                      "The two positions A and B in each output are separated by \";\", "\
                                      "and your output is forbidden to contain extra extraneous characters.
"
    

def use_exp_agent(opt, system_prompt):
    agent = get_bot(engine=opt.engine, api_key=opt.api_key, system_prompt=system_prompt, proxy=opt.net_proxy)
    return agent

def write_replace_instruction(agent, path, caption1, caption2):
    Input = f'1. {caption1} 2. {caption2}'
    Output = get_response(agent, Input)
    with open(path, 'w') as f:
        f.write(Output)
    return Output

def write_move_instruction(agent, path, caption, label, bbox):
    Input = f'{caption}, {label}, {bbox}'
    Output = get_response(agent, Input)
    # Output = Output.split(';')
    # assert len(Output) == 2, f'Output = {Output}'
    # Output = f'{Output[0]}, {Output[1]}'
    with open(path, 'w') as f:
        f.write(Output)
    return Output


