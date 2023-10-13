import cv2
import numpy as np
from matplotlib import pyplot as plt

def perform_segmentation(hair_removed_image):
    hair_removed_gray = cv2.cvtColor(hair_removed_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(hair_removed_gray, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    background_mask = np.zeros_like(thresholded_image)
    background_mask[:30, :] = 255
    thresholded_image_no_triangles = cv2.bitwise_and(thresholded_image, thresholded_image, mask=cv2.bitwise_not(background_mask))

    inverted_image = cv2.bitwise_not(thresholded_image_no_triangles)

    h, w = inverted_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted_image, mask, (0, 0), 0)

    kernel_opening = np.ones((5, 5), np.uint8)
    segmented_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    kernel_smoothing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    segmented_image_smoothed = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel_smoothing, iterations=1)
    segmented_image_smoothed = cv2.morphologyEx(segmented_image_smoothed, cv2.MORPH_CLOSE, kernel_smoothing, iterations=1)

    return segmented_image_smoothed