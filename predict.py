import os
import sys
import torch
from PIL import Image
from torchvision import transforms, models


def load_model(model_path, device):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, model_path)
    model_load = torch.load(model_path, map_location=device, weights_only=True)
    classes = model_load["classes"]
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(model_load["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


def main(image_paths):
    device = torch.device('cpu')
    mean_norm = [0.485, 0.456, 0.406]
    std_norm = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])

    model, classes = load_model('plant_disease_model.pth', device)
    for path in image_paths:
        if os.path.exists(path):
            pred_y = predict_image(model, path, transform, device)
            print(f"Image: {path}, Predicted Class: {classes[str(pred_y)]}")
        else:
            print(f"Error: File {path} does not exist.")


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python predict.py <image_path1> <image_path2> ...")
            sys.exit(1)

        image_paths = sys.argv[1:]
        main(image_paths)
    except Exception as e:
        print(f"Error: {e}")
