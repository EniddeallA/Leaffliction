from plantcv import plantcv as pcv
import numpy as np
import random
from PIL import Image

class PlantCVTransforms:
    def __init__(self, debug = None):
        if debug not in [None, "plot", "print"]:
            raise Exception(f"debug must be None, plot or print; {debug} inserted!")
        pcv.params.debug = debug

    def threshold_binary(self, image, togray = False):
        img = pcv.rgb2gray_lab(rgb_img=image, channel="a") if togray else image
        return pcv.threshold.binary(img, threshold=119, object_type="dark")

    def gaussian_blur(self, image):
        return pcv.gaussian_blur(img=image, ksize=(3, 3))

    def apply_mask(self, image):
        mask = self.threshold_binary(image)
        return pcv.apply_mask(img=image, mask=mask, mask_color='white')

    def shape_analysis(self, image):
        mask = self.threshold_binary(image, True)
        return pcv.analyze.size(img=image, labeled_mask=mask)

    # def x_pseudolandmarks(self, image):
    #     mask = self.threshold_binary(image, True)
    #     np.array([pcv.homology.x_axis_pseudolandmarks(img=image, mask=mask)]).astype(np.float32)

    # def y_pseudolandmarks(self, image):
    #     mask = self.threshold_binary(image, True)
    #     return np.array([pcv.homology.y_axis_pseudolandmarks(img=image, mask=mask)]).astype(np.float32)

    def original_image(self, image):
        return image

    def __call__(self, image):
        image = image if isinstance(image, np.ndarray) else np.array(image)
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]
        random_index = random.randint(0, len(methods) - 1)
        return getattr(self, methods[random_index])(image)

# img_path = "D:\\Projects\\42\\Leaffliction\\images\\Apple_healthy\\image (2).JPG"
# # image = Image.open(img_path).convert('RGB')
# j = PlantCVTransforms()
# image, _, _ = pcv.readimage(filename=img_path)
# j.x_pseudolandmarks(image)