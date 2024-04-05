from socket import *
import numpy as np
import cv2, time
from PIL import Image
from tcputils import receive_image_from_length, Encode_and_Send

def test_MGIE_tcp():
    clientHost, clientPort = '127.0.0.1', 4096
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((clientHost, clientPort))

    while True:
        edit_txt = "replace the dog with a bear."
        # clientSocket.send(str(len(edit_txt)).encode())
        clientSocket.send(edit_txt.encode())
        time.sleep(0.5)
        image = np.array(Image.open('./assets/dog&chair.jpg').convert('RGB'))
        Encode_and_Send(clientSocket, image)
        # time.sleep(2)
        recv_img = receive_image_from_length(clientSocket) # np.array
        cv2.imshow('MGIE Received', cv2.cvtColor(np.array(recv_img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)
        cv2.destroyAllWindows()
        time.sleep(2)


if __name__ == '__main__':
    # Validate_planner_No_Img()
    test_MGIE_tcp()
