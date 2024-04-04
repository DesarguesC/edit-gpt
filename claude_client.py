from prompt.anthropic_util import *
from prompt.guide import get_response
import pandas as pd
import time
from socket import *
serverName = '127.0.0.1'
serverPort = 4003


def main():

    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName, serverPort))

    engine = clientSocket.recv(1024).decode()

    clientSocket.send(f'engine received: \'{engine}\''.encode())
    system_prompt = clientSocket.recv(1024).decode()
    clientSocket.send('system_prompt received'.encode())

    api_key = list(pd.read_csv('key.csv')['key'])[0]
    proxy = 'http://127.0.0.1:7890'

    chatbot = Claude(engine=engine, api_key=api_key, system_prompt=system_prompt, proxy=proxy)

    while True:
        message = clientSocket.recv(4096).decode() # query prompt
        if message is None:
            print('waiting ... ')
            time.sleep(5)
            continue
        response = get_response(chatbot, message, mute_print=True)
        clientSocket.send(response.encode())
        print(f'Msg Sent: [{response}]')

if __name__ == '__main__':
    main()
