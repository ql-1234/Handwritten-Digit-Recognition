import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import Model
import os

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件 {image_path} 不存在！")

    img=Image.open(image_path)
    img_tensor=transform(img)

    return img_tensor.unsqueeze(0)


def predict_digits(model_path,digits_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model=Model().to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    predictions=[]
    for i in range(0,6):
        if i==2:
            continue
        img_path=os.path.join(digits_dir,f"{i}.png")
        input_tensor=preprocess_image(img_path).to(device)

        with torch.no_grad():
            output=model(input_tensor)
            pred=output.argmax(dim=1).item()
        predictions.append(pred)

    return predictions


if __name__ == "__main__":
    current_dir=os.path.dirname(os.path.abspath(__file__))
    model_path=os.path.join(current_dir,"models","lenet-mnist.pkl")
    digits_dir=os.path.join(current_dir,"digits")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")
    if not os.path.isdir(digits_dir):
        raise FileNotFoundError(f"文件夹 {digits_dir} 不存在！")

    digit_predictions=predict_digits(model_path,digits_dir)

    Img_Number=[0, 1, 3, 4, 5]
    i=0
    print("\nPrediction Results:")
    for pred in digit_predictions:
        print(f"Image {Img_Number[i]}.png -> Predicted: {pred} (Expected: {Img_Number[i]})")
        i+=1
