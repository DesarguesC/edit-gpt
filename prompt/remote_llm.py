import re
import time
from socket import *

class LLM_Remote():
    def __init__(self, type="", system_prompt="", claude_model='claude-instant-1.2'):
        self.type = type
        self.system_prompt = system_prompt
        self.host = 4001 if 'imp' in self.type else 4002 if 'llava' in self.type else 4003 # llava -> 4002; imp -> 4001; 4003 -> remote claude
        self.server = '127.0.0.1' # pre-mapped to localhost
        if self.host != 4003:
            self.clientSocket = socket(AF_INET, SOCK_STREAM)
            self.clientSocket.connect((self.server, self.host))
        else:
            # remote pc as a server, ssh-server as a server, but server need to gain answer-messages from client
            self.serverSocket = socket(AF_INET, SOCK_STREAM)
            self.serverSocket.bind((self.server, self.host))
            self.serverSocket.listen(1)
            connectionSocket, addr = self.serverSocket.accept()
            self.connectionSocket = connectionSocket
            print("TCP hand shaking worked.")
            self.connectionSocket.send(claude_model.encode())
            OK = self.connectionSocket.recv(1024).decode()
            print(f'claude_model type sent -> client response: {OK}')
            self.connectionSocket.send(self.system_prompt.encode())
            OK = self.connectionSocket.recv(1024).decode()
            print(f"system_prompt sent -> client response: {OK}")



    # def get_question(self, inputs):
    #     return f'{self.system_prompt}. Here\'s your INPUT: {inputs}.\n[None that no any other characters.]'

    def cut(self, response):
        """
            1. INPUT: ... \nOUTPUT: ...
            2. INPUT: ... OUTPUT: ...
            3. OUTPUT: \n...
            4. ... OUTPUT:
            5. " "
        """
        # find the first "OUTPUT: "
        if response == " " or "OUTPUT" not in response:
            # 5. " "
            return " "
        else:
            res_list = [x.strip().lstrip("\"") for x in re.split(r'OUTPUT', response)]
            string = res_list[1].split('\n')[0]
            for i in range(len(string)):
                if string[i] == ':':
                    return string[i+2:]

        # if response.startswith('OUTPUT: '):
        #     # 3. OUTPUT: \n...
        #     return [x for x in re.split(r'\n|INPUT: ', response) if x not in ['', ' ']][0].lstrip('\"').rstrip('\"')
        # # 1. INPUT: ... \nOUTPUT: ...
        # res_list = [x for x in re.split(r'[\n]', response) if x not in ['', ' ']]
        # print(f'res_list = {res_list}')
        # for x in res_list:
        #     x = x.lstrip("\"")
        #     if 'OUTPUT: ' in x:
        #         return x.lstrip("OUTPUT: ").lstrip("\"")
        # res_list = [x for x in re.split(r'INPUT: ', response) if x not in ['', ' ']]
        # for x in res_list:
        #     if 'OUTPUT: ' in x:
        #         return x.lstrip('OUTPUT: ').lstrip('\"')
        # return ' '

    def ask(self, prompt):
        if 'claude' not in self.type:
            print('-' * 9 + f' Using LLM {self.type.upper()} ' + '-' * 9)
            self.clientSocket.send(prompt.encode())
            # time.sleep(2)
            response = self.clientSocket.recv(4096).decode()
            # self.clientSocket.close()
            print(f'Original Response: {response}')
            response = self.cut(response)
            print(f'Cut Response: {response}')
            return response
        else:
            # for claude
            print('-' * 9 + f' Using Remote {self.type.upper()} ' + '-' * 9)
            self.connectionSocket.send(prompt.encode())
            print(f'Query Message sent. ')
            claude_response_text = self.clientSocket.recv(4096).decode()
            print(f'Response received.')
            return claude_response_text

