import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

def load_model(weight_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    return model

def get_class_names(data_dir):
    class_names = [d.name for d in os.scandir(data_dir) if d.is_dir()]
    class_names.sort()
    return class_names


def test_image(image_path, data_dir=r"D:\Python\BTLXLYA\data_split\train", weight_path=r"D:\Python\BTLXLYA\src\animal_model.pth", threshold=0.7):
    classes = get_class_names(data_dir)

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    if max_prob.item() < threshold:
        return "Unknown"
    return classes[predicted.item()]
