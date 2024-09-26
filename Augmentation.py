import os
from PIL import Image
from torchvision.transforms import v2
import random
import sys


def ImageAugmentation(src: str, img: str):
    image = Image.open(img)
    # on linux
    # imgname = img.split('\')
    # on windows
    imgname = img.split('\\')
    imgname = ('').join(imgname[-1:])

    blur = v2.GaussianBlur(kernel_size=(3, 7),
                           sigma=(1.5, 6))(image)
    blur.save(os.path.join(src,
                           imgname + "_blur.jpg"))

    rotate = v2.RandomRotation(degrees=[-80, 80])(image)
    rotate.save(os.path.join(src,
                             imgname.split('.')[0] + "_rotate.jpg"))

    contrast = v2.ColorJitter(brightness=(0.5, 2),
                              contrast=(0.5, 2),
                              saturation=(0.5, 2))(image)
    contrast.save(os.path.join(src,
                               imgname.split('.')[0] +
                               "_contrast.jpg"))

    scaling = v2.RandomResizedCrop(size=(200, 200))(image)
    scaling.save(os.path.join(src,
                              imgname.split('.')[0] + "_scaling.jpg"))

    shear = v2.RandomAffine(degrees=0, shear=45)(image)
    shear.save(os.path.join(src,
                            imgname.split('.')[0] + '_shear.jpg'))

    if random.randint(0, 1) == 0:
        flip = v2.functional.hflip(image)
    else:
        flip = v2.functional.vflip(image)
    flip.save(os.path.join(src,
                           imgname.split('.')[0] + '_flip.jpg'))


def Augmentation(src: str):
    if src.lower().endswith('.jpg'):
        newsrc = src.split('/')
        newsrc = ('/').join(newsrc[:-1])
        os.makedirs(newsrc + '/Augmented', exist_ok=True)
        ImageAugmentation(newsrc + '/Augmented', src)
    else:
        os.makedirs(src + '/Augmented', exist_ok=True)
        for img in os.listdir(src):
            newsrc = os.path.join(src, img)
            if img.lower().endswith('.jpg'):
                print(newsrc)
                print(src)
                ImageAugmentation(src + '/Augmented', newsrc)


if __name__ == '__main__':
    Augmentation(sys.argv[1])
