import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GridSearchCV


def train_random_forest_classifier(x_train, y_train, x_val, y_val):
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Create a grid search object
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Perform the grid search on the training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters found by the grid search
    best_params = grid_search.best_params_

    # Create a Random Forest classifier with the best parameters
    best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)

    # Train the best classifier on the training data
    best_rf_classifier.fit(x_train, y_train)
    
    # Make predictions on the validation data
    rf_preds = best_rf_classifier.predict(x_val)

    # Calculate accuracy and confusion matrix
    rf_accuracy = accuracy_score(y_val, rf_preds)
    rf_confusion_matrix = confusion_matrix(y_val, rf_preds)
    
    return rf_accuracy, rf_confusion_matrix

def feature_selection_with_random_forest(x_train, y_train, num_features_to_select):
    # Train a Random Forest classifier to rank feature importance
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Sort features by importance
    sorted_feature_indices = np.argsort(feature_importances)[::-1]

    # Select the top 'num_features_to_select' features
    selected_feature_indices = sorted_feature_indices[:num_features_to_select]

    # Extract the selected features
    selected_features = x_train[:, selected_feature_indices]

    return selected_features

def train_svm_classifier_with_feature_selection(x_train, y_train, x_val, y_val, num_features_to_select):
    # Select the top 'num_features_to_select' features
    selected_features = feature_selection_with_random_forest(x_train, y_train, num_features_to_select)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_val)

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
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    return svm_classifier, accuracy, report