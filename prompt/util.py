import torch, cv2
from PIL import Image

def get_image_from_box(image, box):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(image, str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(box)==4, f'box = {box}'
    # print(f'type-box = {type(box)}')
    assert len(image.shape)==3, f'image.shape = {image.shape}'
    x, y, w, h = box[0], box[1], box[2], box[3]
    box_image = image[y:y+h,x:x+w,:]
    # box: x,y,w,h   |   image: h,w,c
    # print(f'box_image.shape = {box_image.shape}')
    return box_image


