import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from segmentation.segmentation import *
from preprocessing.preprocessing import *
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from scipy.stats import moment

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

    segmented_image_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

    # Calculate HOG features for the entire segmented image
    hog_features = hog(segmented_image_gray, pixels_per_cell=(2, 2), cells_per_block=(2, 2))
    # print(f"Feature vector shape after HOG: {hog_features.shape}")

    # Calculate GLCM and extract statistics for the entire segmented image
    glcm = graycomatrix(segmented_image_gray, [1], [0], symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, 'contrast')
    glcm_contrast = glcm_contrast.ravel()
    glcm_energy = graycoprops(glcm, 'energy')
    glcm_energy = glcm_energy.ravel()
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')
    glcm_homogeneity = glcm_homogeneity.ravel()

    # Extract color features for the entire segmented image
    color_features = []

    # Calculate color moments (mean, variance, skewness, kurtosis) for each channel (R, G, B)
    for channel in range(3):
        color_mean = np.mean(channel)
        color_variance = np.var(channel)

        # Calculate skewness and kurtosis
        color_skewness = moment(channel, moment=3)
        color_kurtosis = moment(channel, moment=4)

        # Reshape the variables to have the same shape
        color_mean = np.array([color_mean])
        color_variance = np.array([color_variance])
        color_skewness = np.array([color_skewness])
        color_kurtosis = np.array([color_kurtosis])

        # Extend the color_features array
        color_features.extend([color_mean, color_variance, color_skewness, color_kurtosis])

    # Convert the color_features list to a NumPy array
    color_features = np.array(color_features)

    # Now you can use .ravel() on the NumPy array
    color_features = color_features.ravel()

    # Combine the features into a single feature vector
    feature_vector = np.concatenate([hog_features, glcm_contrast, glcm_energy, glcm_homogeneity, color_features])

    return [feature_vector]  # Return a list with one feature vector for the entire image
