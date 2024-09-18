import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class PlantDiseaseDataset(Dataset):
    authorizedExtentionImages = ['.jpg', '.JPG', '.png', '.jpeg']
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, disease_folder in enumerate(os.listdir(root_dir)):
            disease_folder_path = os.path.join(root_dir, disease_folder)
            if os.path.isdir(disease_folder_path):
                for img_file in os.listdir(disease_folder_path):
                    if any([img_file.endswith(authExt) for authExt in self.authorizedExtentionImages]):
                        self.image_paths.append(os.path.join(disease_folder_path, img_file))
                        self.labels.append(label)
                    else:
                        raise Exception("file extension not supported!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
x = PlantDiseaseDataset("./images")
print(len(x.labels))
print(len(x.image_paths))