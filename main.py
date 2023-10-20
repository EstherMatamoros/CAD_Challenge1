import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from classification.classificationML import train_random_forest_classifier, train_svm_classifier_with_feature_selection, train_qda_classifier
from features.feature_extraction import extract_features_from_segmented_image , convert_to_padded_array, load_image_paths_from_folder
from data_loader.data_loading import load_images_from_folder
from preprocessing.preprocessing import *
import random  

class SkinImageClassifier:
    def __init__(self, nevus_dir, others_dir, val_nevus_dir, val_others_dir):
        self.nevus_dir = nevus_dir
        self.others_dir = others_dir
        self.val_nevus_dir = val_nevus_dir
        self.val_others_dir = val_others_dir
        self.preprocessed_image = None  # Initialize the attribute to None


    def load_preprocessed_image(self):
        if os.path.exists('preprocessed_image.pickle'):
            with open('preprocessed_image.pickle', 'rb') as file:
                preprocessed_image = pickle.load(file)
            self.preprocessed_image = preprocessed_image
        else:
            self.preprocessed_image = None

    def preprocess_image(self, image):
        if self.preprocessed_image is not None:
            return self.preprocessed_image

        resized_image = resize_images(image)
        # Remove hair from the stretched grayscale image
        hair_removed_image = remove_hair(resized_image)

        # Save the preprocessed image to a pickle file
        with open('preprocessed_image.pickle', 'wb') as file:
            pickle.dump(hair_removed_image, file)

        # Set self.preprocessed_image to the processed image
        self.preprocessed_image = hair_removed_image

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
        feature_vector = extract_features_from_segmented_image(image) 
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
            img_features = self.extract_features(img)  # Extract features for the whole image, not regions

            # Append the features and label for the entire image
            features.append(img_features)
            labels.append(label)

        return features, labels
    
    def train_classifier(self, subset_size=None, num_features_to_select=None):
        nevus_features, nevus_labels = self.process_images(self.nevus_dir, label=0, subset_size=subset_size)
        others_features, others_labels = self.process_images(self.others_dir, label=1, subset_size=subset_size)

        val_nevus_features, val_nevus_labels = self.process_images(self.val_nevus_dir, label=0, subset_size=subset_size)
        val_others_features, val_others_labels = self.process_images(self.val_others_dir, label=1, subset_size=subset_size)

        # Combine features and labels for both classes
        train_features = nevus_features + others_features
        train_labels = nevus_labels + others_labels

        val_features = val_nevus_features + val_others_features
        val_labels = val_nevus_labels + val_others_labels

        # After combining features and labels, reshape the features array
        train_features = np.array(train_features)  # Convert to NumPy array
        train_features = train_features.reshape(train_features.shape[0], -1)  # Reshape to 2D

        val_features = np.array(val_features)  # Convert to NumPy array
        val_features = val_features.reshape(val_features.shape[0], -1)  # Reshape to 2D

        print(f"Total number of train features: {len(train_features)}")
        print(f"Total number of  train labels: {len(train_labels)}")

        if len(train_features) == 0:
            print("Error: No features found.")
            return
        
        rf_accuracy, rf_confusion_matrix = train_random_forest_classifier(train_features, train_labels, val_features, val_labels)

        # Print classifier performance metrics
        print(f"Random Forest Accuracy: {rf_accuracy}")
        print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")


        # Now, the features array is 2D, and you can proceed to train the classifier
        svm_scores, svm_accuracy, svm_report = train_svm_classifier_with_feature_selection(train_features, train_labels, val_features, val_labels, num_features_to_select)

        # Print SVM classifier performance metrics
        print(f"SVM Accuracy: {svm_accuracy}")
        print(f"SVM mean CV scores:\n{svm_scores}")
        print(f"SVM Classification Report:\n{svm_report}")

        # Now, the features array is 2D, and you can proceed to train the classifier
        qda_scores, qda_accuracy, qda_report = train_qda_classifier(train_features, train_labels, val_features, val_labels, num_features_to_select)

        # Print SVM classifier performance metrics
        print(f"QDA Accuracy: {qda_accuracy}")
        print(f"QDA mean CV scores:\n{qda_scores}")
        print(f"QDA Classification Report:\n{qda_report}")




if __name__ == "__main__":
    classifier = SkinImageClassifier(nevus_dir='train/train/nevus', others_dir='train/train/others',val_nevus_dir='val/val/val/nevus', val_others_dir='val/val/val/others')
    subset_size = None
    num_features_to_select = 200  # Choose the number of top features to select
    classifier.train_classifier(subset_size=subset_size, num_features_to_select=num_features_to_select)