import re
from socket import *

class LLM_Remote():
    def __init__(self, type="", system_prompt=""):
        self.type = type
        self.system_prompt = system_prompt
        self.host = 4001 if 'imp' in self.type else 4002 # llava -> 4002; imp -> 4001
        self.server = '127.0.0.1' # pre-mapped to localhost
        self.clientSocket = socket(AF_INET, SOCK_STREAM)
        self.clientSocket.connect((self.server, self.host))

    def get_question(self, inputs):
        return f'{self.system_prompt}. Here\'s your INPUT: {inputs}.\n[None that no any other characters.]'

    def cut(self, response):
        re.split(r'OUTPUT: |INPUT: |\n', response)[0].lstrip('\"').rstrip('\"')

    def ask(self, prompt):
        print('-' * 9 + f' Using LLM {self.type.upper()} ' + '-' * 9)
        question = self.get_question(prompt)
        self.clientSocket.send(question.encode())
        response = self.clientSocket.recv(4096).decode()
        print(f'Original Response: {response}')
        response = self.cut(response)
        print(f'Cut Response: {response}')
        return response
