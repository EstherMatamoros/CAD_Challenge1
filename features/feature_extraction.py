import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from segmentation.segmentation import *
from preprocessing.preprocessing import *
from skimage.feature import hog

def load_image_paths_from_folder(folder_path):
    image_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    return image_files

# Convert lists of feature vectors to a NumPy array after padding
def convert_to_padded_array(features):
    # Find the maximum feature vector length
    max_length = max(len(feature) for feature in features)

    # Initialize an empty array filled with zeros
    padded_array = np.zeros((len(features), max_length))

    # Fill the padded array with feature vectors
    for i, feature in enumerate(features):
        # Ensure all feature vectors have the same length
        feature = feature + [0] * (max_length - len(feature))
        padded_array[i] = feature

    return padded_array

def extract_features_from_segmented_image(segmented_image):
    features = []

    # Initialize a combined mask with the same dimensions as the first region
    combined_mask = np.zeros_like(segmented_image[0])

    for region in segmented_image:
        # Ensure that the region has the same dimensions as the combined mask
        if region.shape != combined_mask.shape:
            region = cv2.resize(region, (combined_mask.shape[1], combined_mask.shape[0]))

        # Perform bitwise OR operation to combine the regions
        combined_mask = cv2.bitwise_or(combined_mask, region)

    # Preprocess and extract features from the combined region
    image_noise_removed = cv2.medianBlur(combined_mask, 3)
    image_contrast_stretched = contrast_stretching(image_noise_removed)
    hair_removed_image = remove_hair(image_contrast_stretched)

    # Convert the hair-removed image to grayscale
    hair_removed_image_gray = cv2.cvtColor(hair_removed_image, cv2.COLOR_RGB2GRAY)
    print(f"Feature vector shape: {hair_removed_image_gray.shape}")

    # Check if the image dimensions are too small for HOG
    if hair_removed_image_gray.shape[0] < 8 or hair_removed_image_gray.shape[1] < 8:
        # Handle the case of a very small image (e.g., pad or skip)
        # You can either skip this image:
        # continue
        # Or pad the image:
        print("Skipping small image")
        if hair_removed_image_gray.shape[0] < 8:
            hair_removed_image_gray = np.pad(hair_removed_image_gray, ((0, 8 - hair_removed_image_gray.shape[0]), (0, 0)), mode='constant')
        if hair_removed_image_gray.shape[1] < 8:
            hair_removed_image_gray = np.pad(hair_removed_image_gray, ((0, 0), (0, 8 - hair_removed_image_gray.shape[1])), mode='constant')

    # Apply HOG to the grayscale image
    feature_vector = hog(hair_removed_image_gray, pixels_per_cell=(2, 2), cells_per_block=(2, 2))
    print(f"Feature vector shape after HOG: {feature_vector.shape}")

    features.append(feature_vector)

    return features
