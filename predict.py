import os
import sys
import torch
import matplotlib.pyplot as plt
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


def Display_Result(original_image, transformed_img, prediction):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6),
                            gridspec_kw={'height_ratios': [3, 1]})

    # Display the original image
    axs[0, 0].imshow(original_image)
    axs[0, 0].axis('off')  # Turn off axis
    axs[0, 0].set_title('Original Image')
    # Display the transformed image
    transformed_img = transformed_img.squeeze(0).permute(1, 2, 0).numpy()
    norm_calcul = transformed_img - transformed_img.min()
    norm__calcul = transformed_img.max() - transformed_img.min()
    transformed_img = norm_calcul / norm__calcul
    axs[0, 1].imshow(transformed_img)
    axs[0, 1].axis('off')  # Turn off axis
    axs[0, 1].set_title('Transformed Image')

    # Remove axes from the bottom plots
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')

    # Add the DL classification text in the center
    fig.text(0.5, 0.25, 'DL classification', ha='center',
             fontsize=14, weight='bold')

    # Add the predicted class below
    fig.text(0.5, 0.15, f'Class predicted : {prediction}',
             ha='center', fontsize=12, color='green')

    # Display the plot
    plt.tight_layout()
    plt.show()


def predict_image(model, image_path, transform, device):
    org_image = Image.open(image_path).convert('RGB')
    image = transform(org_image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return org_image, image, predicted.item()


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
            data = predict_image(model, path, transform, device)
            Display_Result(data[0], data[1], classes[str(data[2])])
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
