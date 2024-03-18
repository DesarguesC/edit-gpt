from transformers import ViltProcessor, ViltForQuestionAnswering
import requests, os, json, torch, nltk
from PIL import Image
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')



# preload: load the model and processor
def preload(model_path, device='cuda'):
    # use opt.model_path
    processor = ViltProcessor.from_pretrained(model_path)
    model = ViltForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    return {'procesor': processor, 'model': model} # return model dict

# IsThereExists: return "yes" or "no" if the label exists in the image
def IsThereExists(model_dict, image, label):
    model = model_dict['model']
    processor = model_dict['pcessor']
    text = "Is there a " + label + " in the image?"
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

# IsRemoved: return True if the label is removed from the image
def IsRemoved(model_dict, label, image_ori, image_edited):
    back1 = IsThereExists(model_dict, image_ori, label)
    back2 = IsThereExists(model_dict, image_edited, label)
    if back1 == "yes" and back2 == "no":
        return True
    else:
        return False

# spliting wordsand choose one noun from the text
# using nltk library
def choose_noun(text):
    words = word_tokenize(text)
    pos = nltk.pos_tag(words)
    nouns = [word for word, pos in pos if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    return nouns[0]

# remove test for COCO_val (2017) dataset
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

        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "vilt-b32-finetuned-vqa"
    processor, model = preload(model_path, device)

    result = choose_noun("a cat is in the image")
    print(result)

    result1 = IsRemoved(model, label = "cat", image_ori = Image.open("cat.jpg"), image_edited = Image.open("nocat.jpg"))
    result2 = IsRemoved(model, label = "cat", image_ori = Image.open("cat.jpg"), image_edited = Image.open("cat.jpg"))
    result3 = IsRemoved(model, label = "cat", image_ori = Image.open("nocat.jpg"), image_edited = Image.open("nocat.jpg"))
    print(result1)
    print(result2)
    print(result3)