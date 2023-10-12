import cv2
import os
import numpy as np
from skimage import io, color, feature, measure, segmentation
from skin_binary_clean import *
from matplotlib import pyplot as plt
import pandas as pd

# from skimage.feature import greycomatrix, shannon_entropy

# Load the specific images
def load_specific_img(img_directory, target_filename):
    for filename in os.listdir(img_directory):
        if filename == target_filename:
            image_path = os.path.join(img_directory, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
            return img

def preprocessing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to 767x1022
    image_resized = cv2.resize(img_gray, (600, 600))

    # Noise removal with a 3x3 median filter
    image_noise_removed = cv2.medianBlur(image_resized, 3)

    # Contrast stretching on the grayscale image
    image_contrast_stretched = contrast_stretching(image_noise_removed)

    return image_contrast_stretched

def segmentation(img):

    # Hair removal using bottom-hat filtering
    hair_removed_image = remove_hair(img)

    # Convert the hair-removed image to grayscale
    hair_removed_gray = cv2.cvtColor(hair_removed_image, cv2.COLOR_RGB2GRAY)

    # Apply GaussianBlur to the grayscale image
    blurred = cv2.GaussianBlur(hair_removed_gray, (5, 5), 0)

    # Perform OTSU's thresholding
    _, thresholded_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove background triangles by creating a mask
    background_mask = np.zeros_like(thresholded_image)
    background_mask[:30, :] = 255  # Assuming the triangles are in the top part of the image
    thresholded_image_no_triangles = cv2.bitwise_and(thresholded_image, thresholded_image, mask=cv2.bitwise_not(background_mask))

    # Invert the image to make the skin lesion white and the background black
    inverted_image = cv2.bitwise_not(thresholded_image_no_triangles)

    # Flood fill operation on 4-connected pixels in the background to remove holes
    h, w = inverted_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted_image, mask, (0, 0), 0)

    # Perform morphological opening to remove small objects (fewer than 2000 pixels)
    kernel_opening = np.ones((5, 5), np.uint8)
    segmented_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    # Lesion border smoothening using opening and closing operations
    kernel_smoothing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    segmented_image_smoothed = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel_smoothing, iterations=1)
    segmented_image_smoothed = cv2.morphologyEx(segmented_image_smoothed, cv2.MORPH_CLOSE, kernel_smoothing, iterations=1)

    contours, _ = cv2.findContours(segmented_image_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return segmented_image_smoothed, contours 


def features_within_contour(img, contours):

    feature_list = []

    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        if w > 0 and h > 0:
            # Extract the region of interest (ROI) within the contour
            roi = img[y:y + h, x:x + w]

            # Calculate shape-related features for the contour
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0  # Set circularity to 0 for contours with zero perimeter

            aspect_ratio = w / h

            # Append the features to the list
            feature_list.append([perimeter, area, circularity, aspect_ratio])

    return feature_list

# Example usage:
# img_directory = 'f:\\MAIA\\third semester\\1. CAD\\Challenge two classes\\CAD_Challenge1\\train\\train\\nevus'  # Replace with the actual path to the directory
# target_filename = 'nev00010.jpg'  # Replace with the filename you want to load
# image = load_specific_img(img_directory, target_filename)
# preprocessed_image = preprocessing(image)
# segmented_image, contours = segmentation(preprocessed_image)
# all_features = features_within_contour(preprocessed_image, contours)

# # Print or process the features for each contour as needed
# for i, features in enumerate(all_features):
#     print(f"Contour {i + 1} Features:")
#     for key, value in features.items():
#         print(f"{key}: {value}")

# Optionally, you can save the processed image with the contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# cv2.imwrite('output_image.jpg', image)
