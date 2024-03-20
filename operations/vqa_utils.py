from transformers import ViltProcessor, ViltForQuestionAnswering
import requests, os, json, torch, nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import torch

if not nltk.data.find('taggers/averaged_perceptron_tagger'):
    nltk.download('averaged_perceptron_tagger')

# preload: load the model and processor
def preload_vqa_model(model_path, device='cuda'):
    # use opt.vqa_model_path & opt.device
    if not model_path.endswith('dandelin/vilt-b32-finetuned-vqa'):
        model_path = os.path.join(model_path,'dandelin/vilt-b32-finetuned-vqa')
    processor = ViltProcessor.from_pretrained(model_path)
    model = ViltForQuestionAnswering.from_pretrained(model_path).to(device)
    return {
        'processor': processor, 
        'model': model.to(device)
    }

def How_Many_label(model_dict, image, label, device='cuda'):
    processor = model_dict['processor']
    model = model_dict['model']
    text = "How many" + label + "are in the image?"
    encoding = processor(image, text, return_tensors="pt").to(device)
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return int(model.config.id2label[idx])

def Val_add_amount(model_dict, label, image_ori, image_edited, device='cuda'):
    # return How_Many_label(model, image_edited, label)
    if isinstance(image_edited, list):
        a = How_Many_label(model_dict, image_ori, label, device=device)
        return [(How_Many_label(model_dict, img, label, device=device) - a) for img in image_edited]

    return How_Many_label(model_dict, image_edited, label, device=device) - How_Many_label(model_dict, image_ori, label, device=device)


def choose_noun(text):
    words = word_tokenize(text)
    pos = nltk.pos_tag(words)
    nouns = [word for word, pos in pos if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    return nouns[0]

def IsThereExists(model_dict, image, label, device='cuda'):
    model = model_dict['model']
    processor = model_dict['processor']
    text = "Is there a " + label + " in the image?"
    encoding = processor(image, text, return_tensors="pt").to(device)
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

def remove_test_coco():
    if os.path.exists('autodl-pub/COCO2017'):
        print('COCO_val (2017) dataset found')

    # saving captions_val2017.json to a variable
    captions_val2017 = json.load(open('../autodl-pub/COCO2017/annotations/captions_val2017.json'))

    # iterate every image in the COCO_val (2017) dataset
    for img in os.listdir('../autodl-pub/COCO2017/val2017'):
        # deleting front 0s to get the image id
        img_id = img.split('.')[0].lstrip('0')

        # getting the caption of the image
        caption = captions_val2017['annotations'][int(img_id)]['caption']
        noun = choose_noun(caption)
        # saving the result to a list
        result.append = IsRemoved(model, label = noun, image_ori = Image.open('../autodl-pub/COCO2017/val2017/' + img), image_edited = Image.open('../autodl-pub/(我是edit结果文件夹)/' + img))

def IsRemoved(model_dict, label, image_ori, image_edited, device='cuda'):
    back1 = IsThereExists(model_dict, image_ori, label, device=device)
    if not isinstance(image_edited, list):
        back2 = IsThereExists(model_dict, image_edited, label)
        return (back1 == "yes" and back2 == "no")
    else:
        return [int(back1 == "yes" and IsThereExists(model_dict, img_, label) == "no") for img_ in image_edited]

def A_IsReplacedWith_B(model_dict, label_A, label_B, image_ori, image_edited, device='cuda'):
    print(f'VAQ | labelA = {label_A}, labelB = {label_B}')
    if not isinstance(image_edited, list):
        A_ori = How_Many_label(model_dict, image_ori, label_A, device=device)
        B_ori = How_Many_label(model_dict, image_ori, label_B, device=device)
        A_edited = How_Many_label(model_dict, image_edited, label_A, device=device)
        B_edited = How_Many_label(model_dict, image_edited, label_B, device=device)
        return int((A_edited==0 or A_edited < A_ori) and B_edited > B_ori)
    else:
        A_ori = How_Many_label(model_dict, image_ori, label_A, device=device)
        B_ori = How_Many_label(model_dict, image_ori, label_B, device=device)
        A_edited = [How_Many_label(model_dict, image_edited_, label_A, device=device) for image_edited_ in image_edited]
        B_edited = [How_Many_label(model_dict, image_edited_, label_B, device=device) for image_edited_ in image_edited]
        return [int((A_edited[i]==0 or A_edited[i] < A_ori) and B_edited[i] > B_ori) for i in range(len(A_edited))]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model_path = "../autodl-tmp/vilt-b32-finetuned-vqa"
    processor, model = preload_vqa_model(model_path, device)

    result = Val_add_amount(model, "cat", image_ori = Image.open("cat.png"), image_edited = Image.open("nocat.jpg"))

    print(result)