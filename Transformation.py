import sys
import os
from plantcv import plantcv as pcv


def gaussian_blur(mask):
    pcv.gaussian_blur(img=mask, ksize=(3, 3))


def apply_mask(image, mask):
    pcv.apply_mask(img=image, mask=mask, mask_color='white')


def rect_roi(image):
    pcv.roi.rectangle(img=image, x=35, y=8, h=235, w=200)


def shape_analysis(image, mask):
    pcv.analyze.size(img=image, labeled_mask=mask)


def x_pseudolandmarks(image, mask):
    pcv.homology.x_axis_pseudolandmarks(img=image, mask=mask)


def y_pseudolandmarks(image, mask):
    pcv.homology.y_axis_pseudolandmarks(img=image, mask=mask)


def Transformation(src: str, dest: str = None):
    if src.lower().endswith('.jpg'):
        pcv.params.debug = "plot"
        image, _, _ = pcv.readimage(filename=src)

        # Identify the grayscale image that maximizes
        # the gray value difference between the plant
        # and the background with:
        # cs = pcv.visualize.colorspaces(rgb_img=image,
        #                            original_img=False)
        # We deduce that the 'LAB' colorspace's channel
        # 'a' is the best option for our grayscale image
        grayscale_img = pcv.rgb2gray_lab(rgb_img=image, channel="a")

        # Identify plant from background, by converting
        # the grayscale image to a binary image (MASK)
        # using a threshold value of '119' which is
        # the minimum value between the two peaks in
        # our histogram:
        # hist = pcv.visualize.histogram(img=grayscale_img,
        #                               bins=30)
        s_thresh = pcv.threshold.binary(grayscale_img,
                                        threshold=119,
                                        object_type="dark")
        # Apply Transformations
        gaussian_blur(s_thresh)
        apply_mask(image, s_thresh)
        rect_roi(image)
        shape_analysis(image, s_thresh)
        x_pseudolandmarks(image, s_thresh)
        y_pseudolandmarks(image, s_thresh)
    else:
        pcv.params.debug = "print"
        if dest:
            pcv.params.debug_outdir = dest
            os.makedirs(f"{dest}", exist_ok=True)
        else:
            pcv.params.debug_outdir = f"{src}/Transformed"
            os.makedirs(f"{src}/Transformed", exist_ok=True)
        for img in os.listdir(src):
            if img.lower().endswith('.jpg'):
                image, _, _ = pcv.readimage(
                    filename=os.path.join(src, img))
                grayscale_img = pcv.rgb2gray_lab(rgb_img=image,
                                                 channel="a")
                s_thresh = pcv.threshold.binary(grayscale_img,
                                                threshold=119,
                                                object_type="dark")
                gaussian_blur(s_thresh)
                apply_mask(image, s_thresh)
                rect_roi(image)
                shape_analysis(image, s_thresh)
                x_pseudolandmarks(image, s_thresh)
                y_pseudolandmarks(image, s_thresh)

    pcv.outputs.save_results(filename="result.json",
                             outformat="json")


if __name__ == '__main__':
    Transformation(sys.argv[1])
