import os
import random
import json
import sys
from PIL import Image
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    MinRetreiveImagesLen = 500
    authExImgs = list(Image.registered_extensions().keys())

    def __init__(self, root_dir, total_images=sys.maxsize, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = {}
        self.imageperclass = {}

        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "data.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as json_file:
                data = json.load(json_file)
            if data["root"] == self.root_dir:
                if len(data["image_paths"]) == total_images:
                    print("fetching images from privious launch!")
                    self.image_paths = data["image_paths"]
                    self.labels = data["labels"]
                    self.classes = data["classes"]
                    return
        self.fetch_images()
        self.balance_data(total_images, data_path)

    def fetch_images(self):
        print("fetching images...")

        def assignImages(folder_path, image_paths: list):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if not os.path.isdir(file_path):
                    def norm(file, ex):
                        return file.lower().endswith(ex)
                    if any([norm(file, ex) for ex in self.authExImgs]):
                        image_paths.append(file_path)
                    else:
                        print(f"Warning: {file} file"
                              " extension not supported. Skipping.")
                else:
                    assignImages(os.path.join(folder_path, file), image_paths)

        for label, disease_folder in enumerate(os.listdir(self.root_dir)):
            disease_folder_path = os.path.join(self.root_dir, disease_folder)
            dict = {"label": label, "image_paths": []}
            self.imageperclass[disease_folder] = dict
            self.classes[str(label)] = disease_folder
            if os.path.isdir(disease_folder_path):
                img_paths = self.imageperclass[disease_folder]["image_paths"]
                assignImages(disease_folder_path, img_paths)

    def balance_data(self, total_images, data_path):
        print("balancing data!")
        normlen = self.MinRetreiveImagesLen
        if total_images < normlen:
            msg1 = f"Minimum number of images required is {normlen}."
            raise Exception(f"{msg1} {total_images} is too low!")

        imgvalue = self.imageperclass.values()
        minImg = min(len(data["image_paths"]) for data in imgvalue)
        num_classes = len(self.imageperclass)
        if (minImg * num_classes) < total_images:
            if not total_images == sys.maxsize:
                tot_img = total_images
            else:
                tot_img = "Max size int"
            print(f"Warning: Total images requested: {tot_img}"
                  f", but the maximum available is {minImg * num_classes}.")
            print("Setting up total_images to the lowest"
                  " number of images for balancing")
            total_images = minImg * num_classes

        images_per_class = total_images // num_classes
        rem_img = total_images % num_classes

        for data in self.imageperclass.values():
            num_to_retrieve = images_per_class + (1 if rem_img > 0 else 0)
            rem_img -= 1
            sampled_paths = random.sample(data["image_paths"], num_to_retrieve)
            self.image_paths.extend(sampled_paths)
            self.labels.extend([data["label"]] * num_to_retrieve)

        data = {
            "image_paths": self.image_paths,
            "labels": self.labels,
            "classes": self.classes,
            "root": self.root_dir
                }
        with open(data_path, 'w') as json_file:
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
