import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from classification.classificationML import train_random_forest_classifier
from features.feature_extraction import extract_features_from_segmented_regions
from data_loader.data_loading import load_images_from_folder
from segmentation.segmentation import perform_segmentation
from preprocessing.preprocessing import *

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
                
        resized_image = cv2.resize(image_rgb, (700, 700))

        # Apply contrast stretching
        contrast_stretched_image = contrast_stretching(resized_image)

        # Apply hair removal
        hair_removed_image = remove_hair(contrast_stretched_image)

        return hair_removed_image  # Return the preprocessed image

    def extract_features(self, image):
        # Your feature extraction code here
        feature_vector = extract_features_from_segmented_regions(image)  # Call the imported function
        # Additional feature extraction steps if needed
        return feature_vector

    def process_images(self, class_dir, label):
        # Check if preprocessed images exist in a Pickle file
        pickle_file = f'preprocessed_images_{label}.pickle'
        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as file:
                preprocessed_images = pickle.load(file)
        else:
            images = load_images_from_folder(class_dir)
            preprocessed_images = []

            for img, filename in tqdm(images, desc=f"Processing {label} images", unit="image"):
                preprocessed_image = self.preprocess_image(img)

                # Save the preprocessed images as a Pickle file
                preprocessed_images.append(preprocessed_image)
            
            with open(pickle_file, 'wb') as file:
                pickle.dump(preprocessed_images, file)

        features = []
        labels = []

        for preprocessed_image in preprocessed_images:
            segmented_image = perform_segmentation(preprocessed_image)
            features.extend(self.extract_features(segmented_image))
            labels.extend([label] * len(features))

            # Save the processed image
            output_path = os.path.join(self.output_folder, filename)
            cv2.imwrite(output_path, segmented_image)

        return features, labels

    def train_classifier(self):
        nevus_features, nevus_labels = self.process_images(self.nevus_dir, label=0)
        others_features, others_labels = self.process_images(self.others_dir, label=1)

        # Convert lists of features to NumPy arrays
        nevus_features = np.array(nevus_features)
        others_features = np.array(others_features)

        # Combine features and labels for both classes
        features = np.concatenate([nevus_features, others_features])
        labels = np.concatenate([nevus_labels, others_labels])

        print(f"Total number of features: {len(features)}")
        print(f"Total number of labels: {len(labels)}")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

        if len(features) == 0:
            print("Error: No features found.")
            return

        # Train your classifier on all features and labels
        rf_accuracy, rf_confusion_matrix = train_random_forest_classifier(features, labels)

        # Print classifier performance metrics
        print(f"Random Forest Accuracy: {rf_accuracy}")
        print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")


if __name__ == "__main__":
    classifier = SkinImageClassifier(nevus_dir='train/train/nevus', others_dir='train/train/others')
    classifier.train_classifier()
