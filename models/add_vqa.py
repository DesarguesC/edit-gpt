from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# preload: load the model and processor
def preload(model_path, device='cuda'):
    processor = ViltProcessor.from_pretrained(model_path)
    model = ViltForQuestionAnswering.from_pretrained(model_path)
    return {
        'processor': processor.to(device), 
        'model': model.to(device)
    }

def How_Many_label(model_dict, image, label):
    processor = model_dict['processor']
    model = model_dict['model']
    text = "How many" + label + "are in the image?"
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return int(model.config.id2label[idx])

def Val(model_dict, label, image_ori, image_edited):
    
#     return How_Many_label(model, image_edited, label)
    return How_Many_label(model_dict, image_edited, label) - How_Many_label(model, image_ori, label)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model_path = "vilt-b32-finetuned-vqa"
    processor, model = preload(model_path, device)

    result = Val(model, "cat", image_ori = Image.open("cat.png"), image_edited = Image.open("nocat.jpg"))

    print(result)