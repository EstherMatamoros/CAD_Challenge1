import cv2
import numpy as np
from matplotlib import pyplot as plt

'''def print_image_statistics(image, label):
    print(f"Statistics for {label} Image:")
    print(f"Minimum Value: {np.min(image)}")
    print(f"Mean Value: {np.mean(image)}")
    print(f"Maximum Value: {np.max(image)}")
    print()'''

def contrast_stretching(image):
    avg = np.mean(image)
    std_dev = np.std(image)
    low_in = avg - 0.4 * std_dev
    high_in = avg + 0.4 * std_dev

    stretched_image = np.clip((image - low_in) / (high_in - low_in) * 255, 0, 255).astype(np.uint8)

    return stretched_image

def remove_hair(image):
    # Ensure the input image is a 3-channel color image
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_hair = np.array([0, 20, 70], dtype=np.uint8)
    upper_hair = np.array([20, 255, 255], dtype=np.uint8)
    mask_hair = cv2.inRange(hsv, lower_hair, upper_hair)
    kernel = np.ones((15, 15), np.uint8)
    bottom_hat = cv2.morphologyEx(image_rgb, cv2.MORPH_BLACKHAT, kernel)
    result_hair_removed = cv2.add(image_rgb, bottom_hat)
    kernel = np.ones((15, 15), np.uint8)
    result_hair_removed = cv2.morphologyEx(result_hair_removed, cv2.MORPH_OPEN, kernel, iterations=2)
    return result_hair_removed