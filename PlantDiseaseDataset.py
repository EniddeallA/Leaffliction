import os, random, json, sys
from PIL import Image
from torch.utils.data import Dataset

class PlantDiseaseDataset(Dataset):
    MinRetreiveImagesLen = 500
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