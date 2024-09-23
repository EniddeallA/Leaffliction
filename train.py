import os, random, json, sys
from PIL import Image
import torch.utils
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Transformation import PlantCVTransforms

class PlantDiseaseDataset(Dataset):
    MinRetreiveImagesLen = 100
    authorizedExtentionImages = ['.jpg', '.JPG', '.png', '.jpeg']
    def __init__(self, root_dir, total_images=sys.maxsize, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = {}
        self.imageperclass = {}

        if os.path.exists("data.json"):
            with open('data.json', 'r') as json_file:
                data = json.load(json_file)
            if data["root"] == self.root_dir and len(data["image_paths"]) == total_images:
                print("fetching images from privious launch!")
                self.image_paths = data["image_paths"]
                self.labels = data["labels"]
                self.classes = data["classes"]
                return
        self.fetch_images()
        self.balance_data(total_images)


    def fetch_images(self):
        print("fetching images...")
        def assignImages(folder_path, image_paths : list):
            for img_file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, img_file)
                if not os.path.isdir(file_path):
                    if any([img_file.endswith(authExt) for authExt in self.authorizedExtentionImages]):
                        image_paths.append(file_path)
                    else:
                        print(f"Warning: {img_file} file extension not supported. Skipping.")
                else:
                    assignImages(os.path.join(folder_path, img_file), image_paths)

        for label, disease_folder in enumerate(os.listdir(self.root_dir)):
            disease_folder_path = os.path.join(self.root_dir, disease_folder)
            self.imageperclass[disease_folder] = {"label":label, "image_paths":[]}
            self.classes[label] = disease_folder
            if os.path.isdir(disease_folder_path):
                assignImages(disease_folder_path, self.imageperclass[disease_folder]["image_paths"])

    def balance_data(self, total_images):
        print("balancing data!")
        if total_images < self.MinRetreiveImagesLen:
            raise Exception(f"Minimum number of images required is {self.MinRetreiveImagesLen}. {total_images} is too low!")

        min_images_per_class = min(len(data["image_paths"]) for data in self.imageperclass.values())
        num_classes = len(self.imageperclass)
        if (min_images_per_class * num_classes) < total_images:
            print(f"Warning: Total images requested: {total_images if not total_images == sys.maxsize else "Max size int"}, but the maximum available is {min_images_per_class * num_classes}.")
            print("Setting up total_images to the lowest number of images for balancing")
            total_images = min_images_per_class * num_classes

        images_per_class = total_images // num_classes
        remaining_images = total_images % num_classes

        for data in self.imageperclass.values():
            num_to_retrieve = images_per_class + (1 if remaining_images > 0 else 0)
            remaining_images -= 1
            sampled_paths = random.sample(data["image_paths"], num_to_retrieve)
            self.image_paths.extend(sampled_paths)
            self.labels.extend([data["label"]] * num_to_retrieve)

        data = {"image_paths":self.image_paths, "labels":self.labels, "classes":self.classes, "root":self.root_dir}
        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self._get_conv_output_size(), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def split_train_data(dataset : PlantDiseaseDataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader

def fit_model(model : CustomCNN, train_loader : DataLoader, val_loader : DataLoader, num_epochs : int,
            device : torch.device, criterion : nn.CrossEntropyLoss, optimizer : optim.Adam):

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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

        val_accuracy = 100.0 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'custom_cnn_plant_disease_model.pth')
    print("Training complete. Model saved.")

if __name__ == "__main__":
    try:
        plantcv_transform = PlantCVTransforms()
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Lambda(lambda img: plantcv_transform(img)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(0.1),
            transforms.r(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = PlantDiseaseDataset("./images", total_images=100, transform=train_transform)
        print("splitting data!")
        train_loader, val_loader = split_train_data(dataset)
        print("setting up model!")
        model = CustomCNN(len(dataset.classes))
        print("setting up device!")
        device = torch.device("cpu")
        model = model.to(device)
        print("setting up loss function")
        criterion = nn.CrossEntropyLoss()
        print("setting up optimizer")
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        print("Training begging")
        fit_model(model, train_loader, val_loader, 10, device, criterion, optimizer)
    except Exception as e:
        print(f"Error: {e}")
