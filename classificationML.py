import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Define your data directory paths for 'nevus' and 'other' images
nevus_dir = 'train\\train\\nevus'
other_dir = 'train\\train\\others'

# Load and preprocess images
def load_and_preprocess_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  # You can adjust the file format as needed
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (128, 128))  # Resize the images to a common size
            images.append(img)
            labels.append(directory.split('_')[-1])  # Assuming folder names indicate class labels
    return np.array(images), np.array(labels)

nevus_images, nevus_labels = load_and_preprocess_images(nevus_dir)
other_images, other_labels = load_and_preprocess_images(other_dir)

# Concatenate and label the data
X = np.concatenate((nevus_images, other_images))
y = np.concatenate((nevus_labels, other_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Flatten the images for RandomForestClassifier and LDA
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Train the Random Forest and LDA classifiers on the original data
rf_classifier.fit(X_train_rf, y_train)
rf_preds = rf_classifier.predict(X_test_rf)

# Calculate accuracy for all classifiers
rf_accuracy = accuracy_score(y_test, rf_preds)

# Calculate confusion matrices for all classifiers
rf_confusion_matrix = confusion_matrix(y_test, rf_preds)


print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")
