import cv2
import numpy as np
from PIL import Image

def receive_from_length(socks, length):
        buf = b''
        while length > 0:
            new_buf = socks.recv(length)
            if not new_buf:
                return buf
            buf = buf + new_buf
            length = length - len(new_buf)
        return buf

def receive_image_from_length(socks, length=16):
    length = receive_from_length(socks, length)
    img_str = receive_from_length(socks, int(length))
    data = np.fromstring(img_str, dtype='uint8')
    decode_img = cv2.imdecode(data, 1)
    return Image.fromarray(decode_img).convert('RGB')

def Encode_and_Send(socks, img_Image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    result, img_encode = cv2.imencode('.jpg', np.array(img_Image), encode_param)
    str_data = np.array(img_encode).tostring()
    socks.send(str(len(str_data)).ljust(16).encode())
    print('length sent.')
    socks.send(str_data)
    print('image sent.')