####################################
#   Put it in MILVLG/IMP           #
#   to start remote LLM server     #
####################################

import sys, os, torch, requests
import os.path as osp
from socket import *
from time import sleep
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import time 

# file_path = './student.csv'
# server_host = socket.gethostname()
server_host = '127.0.0.1'
serverPort = 4001

planning_system_prompt = "You are an image editing system that can give editing solutions based on only 5 editing tools. "\
                            "You have and only have the following 5 types of tools for editing: 'Add', 'Remove', 'Replace', 'Move' and 'Transfer'. "\
                            "The commands are explained and their effects are described below. "\
                            "For the input instructions, "\
                            "you need to give the editing tool use plan according to the overall editing requirements of the image. "\
                            "The tasks of each step are specified in order in the form of $(type, method)$item, "\
                            "Pay attention to the items between the \";\" Separate. Here are two examples of input and output. \n"\
                            "INPUT: \"A women enters the livingroom and take the box on the desk, while a cuckoo flies into the house. And replace the desk with a red chair.\n"\
                            "OUTPUT: \"(Remove, \"remove the box on the desk\");  (Add, \"add a cukoo in the house\"); (Replace, \"Replace the desk with a red chair\"). \n"\
                            "A pair of parentheses with only a \"type\" and an \"edit instruction\", "\
                            "you mustn\'t output any othesr character. "\
                            "Here's your INPUT: "

notes = ". [Note that no any other characters.]"






serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind((server_host, serverPort))
serverSocket.listen(10)
    
model = AutoModelForCausalLM.from_pretrained(
        "../imp-v1-3b", 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

tokenizer = AutoTokenizer.from_pretrained("../imp-v1-3b", local_files_only=True, trust_remote_code=True)


def Tcplink(sock, addr):
    # print('Accept new connection form %s:%s...' % addr)
    # sock.send(b'Welcome!')
    print(f'IMP is ready to serve on \'{server_host}:{serverPort}\'')
    while True:
        message = sock.recv(4096).decode()
        print(f'Message Reveived: {message}')
        input_ids = tokenizer(f'{planning_system_prompt} {message}. {notes}', return_tensors='pt').input_ids.to('cuda')
        output_ids = model.generate(
            input_ids,
            temperature=0.8,
            max_new_tokens=150,
            images=None,
            use_cache=True
        )[0]
        outputs = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        if outputs == '':
            outputs = " "
        print(f'outputs = [{outputs}]')
        sock.send(outputs.encode())
        # sleep(0.5)
        print('finish')


def main():
    
    
    while True:
        connectionSocket, addr = serverSocket.accept()
        #  创建新线程来处理连接
        t = threading.Thread(target=Tcplink(connectionSocket, addr), args=(connectionSocket, addr))
        t.start()

    serverSocket.close()
    sys.exit()
    #Terminate the program after sending the corresponding data

if __name__ == '__main__':
    main()


