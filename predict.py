import os, json, sys, torch
from PIL import Image
from torchvision import transforms, models

def load_model(model_path, num_classes, device):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

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
    
    # Load saved data to get class labels
    with open('data.json', 'r') as f:
        data = json.load(f)
    classes = data['classes']

    # Define transform (same as used in training)
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load the trained model
    model = load_model('plant_disease_model.pth', len(classes), device)
    
    # Predict for each image
    for image_path in image_paths:
        if os.path.exists(image_path):
            predicted_class = predict_image(model, image_path, transform, device)
            print(f"Image: {image_path}, Predicted Class: {classes[str(predicted_class)]}")
        else:
            print(f"Error: File {image_path} does not exist.")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python predict.py <image_path1> <image_path2> ...")
            sys.exit(1)

        image_paths = sys.argv[1:]
        main(image_paths)
    except Exception as e:
        print(f"Error: {e}")
