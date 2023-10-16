import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# def load_and_preprocess_images(directory):
#     images = []
#     labels = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".jpg"):
#             img = cv2.imread(os.path.join(directory, filename))
#             img = cv2.resize(img, (128, 128))
#             images.append(img)
#             labels.append(directory.split('_')[-1])
#     return np.array(images), np.array(labels)

def train_random_forest_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)
    rf_preds = rf_classifier.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_confusion_matrix = confusion_matrix(y_test, rf_preds)

    return rf_accuracy, rf_confusion_matrix