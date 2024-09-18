import matplotlib.pyplot as plt
import os
from torchvision.transforms import v2
from PIL import Image
import random

def fetch_image_dataset(path:str):
    apple = {}
    grape = {}
    for subd in os.listdir(path):
        # inside images
        if subd.startswith('Apple'):
            # apple = {'apple_healthy': [], 'apple_black_rot': [], 'apple_cedar_rust': [], 'apple_scab': []}
            apple[subd] = []
            for img in os.listdir(os.path.join(path,subd)):
                apple[subd].append(img)
        elif subd.startswith('Grape'):
            # grape = {'grape_healthy': [], 'grape_black_rot': [], 'apple_esca': [], 'apple_spot': []}
            grape[subd] = []
            for img in os.listdir(os.path.join(path,subd)):
                grape[subd].append(img)
        else:
            print('Plant not supported')
    return apple, grape

apple, grape = fetch_image_dataset('./images')

def analyze_distribution(data:dict):
    dist = {}
    for k,v in data.items():
        dist[k] = len(v)
    return dist

apple_dist = analyze_distribution(apple)
grape_dist = analyze_distribution(grape)

def plot_distribution(data: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define color palette
    colors = plt.cm.tab10.colors

    # Bar chart
    ax1.bar(data.keys(), data.values(), color=colors[:len(data)])
    # Pie chart
    ax2.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', colors=colors[:len(data)])
    
    plt.title(f'{list(data.keys())[0][:5]} Distribution')
    plt.tight_layout()
    plt.show()
    
plot_distribution(apple_dist)
plot_distribution(grape_dist)

def Augmentation(path:str):
    for subd in os.listdir(path):
        newpath = os.path.join(path, subd)
        os.makedirs(newpath + '/Augmented', exist_ok=True)
        for img in os.listdir(newpath):
            if img.lower().endswith('.jpg'):
                image = Image.open(os.path.join(newpath +'/', img))

                blur = v2.GaussianBlur(kernel_size=(3, 7), sigma=(1.5, 6))(image)
                blur.save(os.path.join(newpath + '/Augmented', img.split('.')[0] + "_blur.jpg"))

                rotate = v2.RandomRotation(degrees=[-80, 80])(image)
                rotate.save(os.path.join(newpath + '/Augmented', img.split('.')[0] + "_rotate.jpg"))
                
                contrast = v2.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2), saturation=(0.5, 2))(image)
                contrast.save(os.path.join(newpath + '/Augmented', img.split('.')[0] + "_contrast.jpg"))
                
                scaling = v2.RandomResizedCrop(size=(200, 200))(image)
                scaling.save(os.path.join(newpath + '/Augmented', img.split('.')[0] + "_scaling.jpg"))
                
                shear = v2.RandomAffine(degrees=0, shear=45)(image)
                shear.save(os.path.join(newpath + '/Augmented', img.split('.')[0] + '_shear.jpg'))
                
                if random.randint(0, 1) == 0:
                    flip = v2.functional.hflip(image)
                else:
                    flip = v2.functional.vflip(image)
                flip.save(os.path.join(newpath + '/Augmented', img.split('.')[0] + '_flip.jpg'))

path = "./images"
Augmentation(path)

