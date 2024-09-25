import os
import argparse
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, utils
from PlantDiseaseDataset import PlantDiseaseDataset


def save_images(dataset, dataset_type, classes):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_set_path = os.path.join(base_path, dataset_type)
    if os.path.exists(data_set_path):
        shutil.rmtree(data_set_path)
    os.makedirs(data_set_path)
    image_number = {}
    for image, label in dataset:
        label = str(label)
        image_number[label] = image_number.get(label, -1) + 1
        img_path = f'{classes[label]}_image_{image_number[label]}.png'
        utils.save_image(image, os.path.join(data_set_path, img_path))
    print(f"{len(dataset)} images are saved in {data_set_path}.")


def split_train_data(dataset: PlantDiseaseDataset, batch_size=32):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train Data lenght = {len(train_set)}.")
    print(f"Test Data lenght = {len(val_dataset)}.")
    save_images(dataset, 'preprocess_images', dataset.classes)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def fit_model(model: models.ResNet, train_loader: DataLoader,
              val_loader: DataLoader, num_epochs: int,
              device: torch.device, criterion: torch.nn.CrossEntropyLoss,
              optimizer: torch.optim.Adam):

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}],"
              f"Loss: {running_loss/len(train_loader):.4f},"
              f" Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, val_labels in val_loader:
                images, val_labels = images.to(device), val_labels.to(device)
                val_outputs = model(images)
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

        val_accuracy = 100.0 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

    return model


def save_model(model: models.RegNet, classes):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'plant_disease_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes
    }, model_path)


def get_args():
    parser = argparse.ArgumentParser(description="Train a leaves module.")
    parser.add_argument("--data_train", type=str,
                        required=True, help="Path to the datatrain folder")
    parser.add_argument("--total_images", type=int,
                        default=1000, help="Total Images to be retreived!")

    args = parser.parse_args()
    return args.data_train, args.total_images


if __name__ == "__main__":
    try:
        mean_norm = [0.485, 0.456, 0.406]
        std_norm = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])
        root_dir, total_images = get_args()
        kwargs = {
            "root_dir": root_dir,
            "total_images": total_images,
            "transform": train_transform
        }
        dataset = PlantDiseaseDataset(**kwargs)
        print("splitting data!")
        train_loader, val_loader = split_train_data(dataset)
        print("setting up model!")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(dataset.classes))
        print("setting up device!")
        device = torch.device("cpu")
        model = model.to(device)
        print("setting up loss function")
        criterion = torch.nn.CrossEntropyLoss()
        print("setting up optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                     weight_decay=1e-5)
        # print("Training begging")
        # model = fit_model(model, train_loader,
        #                   val_loader, 6, device, criterion, optimizer)
        # print("Saving Model data.")
        # save_model(model, dataset.classes)
    except Exception as e:
        print(f"Error: {e}")
