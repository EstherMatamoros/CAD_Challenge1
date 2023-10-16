import cv2
import numpy as np
from matplotlib import pyplot as plt
from segmentation.segmentation import *
from preprocessing.preprocessing import *
from skimage.feature import hog


# Extract features from segmented regions using the binary mask obtained from segmentation
def extract_features_from_segmented_regions(original_image):
    # Perform image segmentation to obtain a binary mask
    segmented_image_smoothed = perform_segmentation(original_image)

    features = []
    h, w, _ = original_image.shape

    for region in segmented_image_smoothed:
        # Ensure the mask is the same size as the original image
        if region.shape[:2] != (h, w):
            # Resize the region mask to match the original image dimensions
            region = cv2.resize(region, (w, h))
            region = np.uint8(region)

            # Use the binary mask to extract the region from the original image
        region_of_interest = cv2.bitwise_and(original_image, original_image, mask=region)

        # Preprocess and extract features from the region_of_interest
        # Contrast stretching on the grayscale image
        image_noise_removed = cv2.medianBlur(region_of_interest, 3)
        image_contrast_stretched = contrast_stretching(image_noise_removed)
        hair_removed_image = remove_hair(image_contrast_stretched)
        feature_vector = hog(hair_removed_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features.append(feature_vector)
    return features