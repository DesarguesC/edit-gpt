####################################
#   Put it in LLaVA                #
#   to start remote LLM server     #
####################################
import sys, os, torch, requests
import os.path as osp
from socket import *
from time import sleep
import threading
import time 

from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
from io import BytesIO
import re


# file_path = './student.csv'
# server_host = socket.gethostname()
server_host = '127.0.0.1'
serverPort = 4002

planning_system_prompt = "You are an image editing system that can give editing solutions based on only 5 editing tools. "\
                            "You have and only have the following 5 types of tools for editing: 'Add', 'Remove', 'Replace', 'Move' and 'Transfer'. "\
                            "The commands are explained and their effects are described below. "\
                            "\'Add\' can add objects, such as \"add an apple \",\" put two teddy bears on the table \"; "\
                            "\'Remove\' can be used to remove objects, e.g. \'Remove a pear from a table\'; "\
                            "\'A person has taken the broom away\'; \'Replace\' is used to replace an object, "\
                            "such as \"replace a lion with a tiger \", \" replace the moon with the sun \"; "\
                            "\'Move\' is used to move something, as in \'move the coffee from the left side of "\
                            "the computer to the right side\'; \'Transfer\' is used for style transfer, e.g. "\
                            "\'modernist style\', \'to Renaissance style\'. For the input instructions, "\
                            "you need to give the editing tool use plan according to the overall editing requirements of the image. "\
                            "The tasks of each step are specified in order in the form of $(type, method)$item. "\
                            "Pay attention to the items between the \";\" Separate. Here are two examples of input and output. \n"\
                            "INPUT: \"A women enters the livingroom and take the box on the desk, while a cuckoo flies into the house. And replace the desk with a red chair.\n"\
                            "OUTPUT: \"(Remove, \"remove the box on the desk\");  (Add, \"add a cukoo in the house\"); (Replace, \"Replace the desk with a red chair\"). \n"\
                            "INPUT: \"The sun went down, the sky was suddenly dark, and the birds returned to their nests. \"\n"\
                            "Output: (Remove, \"remove the sun\"); (Transfer, \"the lights are out, darkness\"); "\
                            "(Add, \"add some birds, they are flying in the sky\")\nNote that when you are giving output, \n"\
                            "A pair of parentheses with only a \"type\" and an \"edit instruction\", "\
                            "you mustn\'t output any other character. "\
                            "Here's your INPUT: "

notes = ". [Note that no any other characters.]"


serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind((server_host, serverPort))
serverSocket.listen(10)

model_path = "../autodl-tmp/llava-v1.6-vicuna-7b"
conv_mode = "llava_v0"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
    
 
 
def Tcplink(sock, addr):
    # print('Accept new connection form %s:%s...' % addr)
    # sock.send(b'Welcome!')
    print(f'LLaVA-1.5 is ready to serve on \'{server_host}:{serverPort}\'')
    while True:
        message = sock.recv(4096).decode()
        print(f'Message Reveived: {message}')
        input_ids = (
            tokenizer_image_token(f'{planning_system_prompt} {message}. {notes}', tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        output_ids = model.generate(
            input_ids,
            images=None,
            image_sizes=None,
            do_sample=True,
            temperature=0.8,
            top_p=0.,
            num_beams=1,
            max_new_tokens=100,
            use_cache=True,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs == '':
            outputs = " "
        print(f'outputs = [{outputs}]')
        sock.send(outputs.encode())
        # sleep(0.5)
        print('finish')
 

# TCP server
def main():
    while True:
        #  接受一个新连接：
        connectionSocket, addr = serverSocket.accept()
        #  创建新线程来处理连接
        t = threading.Thread(target=Tcplink(connectionSocket, addr), args=(connectionSocket, addr))
        t.start()
        # time.sleep(1)
        
    sys.exit()
    #Terminate the program after sending the corresponding data

if __name__ == '__main__':
    main()


