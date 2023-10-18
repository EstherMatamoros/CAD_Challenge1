import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from classification.classificationML import train_random_forest_classifier, train_svm_classifier_with_feature_selection
from features.feature_extraction import extract_features_from_segmented_image , convert_to_padded_array, load_image_paths_from_folder
from data_loader.data_loading import load_images_from_folder
from segmentation.segmentation import perform_segmentation
from preprocessing.preprocessing import *
import random  

class SkinImageClassifier:
    def __init__(self, nevus_dir, others_dir):
        self.nevus_dir = nevus_dir
        self.others_dir = others_dir
        self.output_folder = 'CAD_Challenge1/masks'
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_image(self, image):
        # Convert grayscale image to 3-channel color image if needed
        if len(image.shape) == 2 or image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        resized_image = resize_images(image_rgb)
        # Apply contrast stretching
        contrast_stretching_image = contrast_stretching(resized_image)
        # Apply hair removal
        hair_removed_image = remove_hair(contrast_stretching_image)

        return hair_removed_image

    def load_random_subset(self, folder_path, subset_size):
        # List all files in the folder
        all_files = os.listdir(folder_path)

        # Randomly select a subset of files
        random_files = random.sample(all_files, subset_size)

        # Load and preprocess the selected files
        subset_images = []
        for filename in random_files:
            image_path = os.path.join(folder_path, filename)
            preprocessed_image = self.preprocess_image(cv2.imread(image_path))
            subset_images.append(preprocessed_image)

        return subset_images

    def extract_features(self, image):
        # Your feature extraction code here
        feature_vector = extract_features_from_segmented_image(image) 
        # Additional feature extraction steps if needed
        return feature_vector

    def process_images(self, class_dir, label, subset_size=None):
        if subset_size is not None:
            images = self.load_random_subset(class_dir, subset_size)
        else:
            image_files = load_image_paths_from_folder(class_dir)
            images = [self.preprocess_image(cv2.imread(img_path)) for img_path in image_files]

        features = []
        labels = []

        for img in tqdm(images, desc=f"Processing {label} images", unit="image"):
            segmented_image = perform_segmentation(img)
            img_features = self.extract_features(img)  # Extract features for the whole image, not regions

            # Append the features and label for the entire image
            features.append(img_features)
            labels.append(label)

            output_path = os.path.join(self.output_folder, f"{label}_{len(features)}.png")
            cv2.imwrite(output_path, segmented_image)

        return features, labels

    def train_classifier(self, subset_size=None, num_features_to_select=None):
        nevus_features, nevus_labels = self.process_images(self.nevus_dir, label=0, subset_size=subset_size)
        others_features, others_labels = self.process_images(self.others_dir, label=1, subset_size=subset_size)

        # Combine features and labels for both classes
        features = nevus_features + others_features
        labels = nevus_labels + others_labels


        # After combining features and labels, reshape the features array
        features = np.array(features)  # Convert to NumPy array
        features = features.reshape(features.shape[0], -1)  # Reshape to 2D

        print(f"Total number of features: {len(features)}")
        print(f"Total number of labels: {len(labels)}")

        if len(features) == 0:
            print("Error: No features found.")
            return
        
        rf_accuracy, rf_confusion_matrix = train_random_forest_classifier(features, labels)

        # Print classifier performance metrics
        print(f"Random Forest Accuracy: {rf_accuracy}")
        print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")


        # Now, the features array is 2D, and you can proceed to train the classifier
        svm_classifier, svm_accuracy, svm_report = train_svm_classifier_with_feature_selection(features, labels, num_features_to_select = 200)

        # Print SVM classifier performance metrics
        print(f"SVM Accuracy: {svm_accuracy}")
        print(f"SVM Classification Report:\n{svm_report}")




if __name__ == "__main__":
    classifier = SkinImageClassifier(nevus_dir='train/train/nevus', others_dir='train/train/others')
    subset_size = 1000
    num_features_to_select = 200  # Choose the number of top features to select
    classifier.train_classifier(subset_size=subset_size, num_features_to_select=num_features_to_select)