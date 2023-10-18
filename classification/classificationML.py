import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import IncrementalPCA



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

def feature_selection_with_random_forest(features, labels, num_features_to_select):
    # Train a Random Forest classifier to rank feature importance
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(features, labels)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Sort features by importance
    sorted_feature_indices = np.argsort(feature_importances)[::-1]

    # Select the top 'num_features_to_select' features
    selected_feature_indices = sorted_feature_indices[:num_features_to_select]

    # Extract the selected features
    selected_features = features[:, selected_feature_indices]

    return selected_features

def train_svm_classifier_with_feature_selection(features, labels, num_features_to_select):
    # Select the top 'num_features_to_select' features
    selected_features = feature_selection_with_random_forest(features, labels, num_features_to_select)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform Incremental PCA
    n_components = 50  # Choose the desired number of components
    ipca = IncrementalPCA(n_components=n_components, batch_size=100)  # Adjust the batch size according to available memory
    X_train_pca = ipca.fit_transform(X_train_scaled)
    X_test_pca = ipca.transform(X_test_scaled)

    # Create an SVM classifier with the RBF (Medium Gaussian) kernel
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Train the SVM classifier
    svm_classifier.fit(X_train_pca, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test_pca)

    # Calculate accuracy and generate a classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return svm_classifier, accuracy, report