import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize_images(image, new_size=(227, 227), preserve_ratio=False):
    '''Resizing function to handle image sizes from different datasets.
    It can resize to a fixed size or preserve the aspect ratio.
    '''
    if preserve_ratio:
        # If you want to preserve the aspect ratio, you can use code like this:
        # Compute the aspect ratio of the input image
        height, width, _ = image.shape
        aspect_ratio = width / height

        # Set the dimensions based on the target width while preserving the aspect ratio
        target_width = new_size[0]
        target_height = int(target_width / aspect_ratio)
        target_size = (target_width, target_height)

        # Resize the image
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    else:
        # Resize to the fixed size
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    return resized_image


def remove_hair(image):

    # Convert the resized image to grayscale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # Adjust kernel size

    # Perform the bottom hat filtering on the grayscale image
    bottom_hat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)

    # Intensify the hair contours in preparation for the inpainting algorithm
    _, thresh2 = cv2.threshold(bottom_hat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the resized image depending on the mask
    inpainted_resized = cv2.inpaint(image, thresh2, 1, cv2.INPAINT_TELEA)

    # Resize the inpainted image back to the original size
    result_hair_removed = cv2.resize(inpainted_resized, (image.shape[1], image.shape[0]))

    return result_hair_removed
