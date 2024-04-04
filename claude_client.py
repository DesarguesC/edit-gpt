from prompt.anthropic_util import *
from prompt.guide import get_response
import pandas as pd
from socket import *
serverName = '127.0.0.1'
serverPort = 4003


def main():

    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName, serverPort))

    engine = clientSocket.recv(1024).decode()
    clientSocket.send('engine received'.encode())
    system_prompt = clientSocket.recv(1024).decode()
    clientSocket.send('system_prompt received'.encode())

    api_key = list(pd.read_csv('key.csv')['key'])[0]
    proxy = 'http://127.0.0.1:7890'

    chatbot = Claude(engine=engine, api_key=api_key, system_prompt=system_prompt, proxy=proxy)

    while True:
        message = clientSocket.recv(4096).decode() # query prompt
        response = get_response(chatbot, message, mute_print=True)
        print(f'Msg Received from Claude LLM: {response}')
        clientSocket.send(response.encode())

if __name__ == '__main__':
    main()
