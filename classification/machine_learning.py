import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from features.features_from_contours import * 
from sklearn.svm import SVC

def preprocess_and_extract_features(dataset_directory):
    feature_list = []
    labels = []

    for class_name in os.listdir(dataset_directory):
        class_directory = os.path.join(dataset_directory, class_name)

        # Check if it's a directory before processing
        if os.path.isdir(class_directory):
            for filename in os.listdir(class_directory):
                image_path = os.path.join(class_directory, filename)
                img = cv2.imread(image_path)

                if img is not None:
                    img = preprocessing(img)
                    segmented_img, contours = segmentation(img)
                    features_df = features_within_contour(segmented_img, contours)

                    # Check if features are not None
                    if features_df is not None and not features_df.empty:
                        # Determine the label based on the directory structure
                        if "nevus" in class_name.lower():
                            label = 1  # Positive label for nevus
                        elif "others" in class_name.lower():
                            label = 0  # Negative label for others
                        else:
                            continue  # Skip images in other folders

                        feature_list.append(features_df)  # Append DataFrame
                        labels.extend([label] * len(features_df))

    # Concatenate the list of DataFrames into a single DataFrame
    if feature_list:
        feature_df = pd.concat(feature_list, ignore_index=True)
    else:
        feature_df = pd.DataFrame()  # Empty DataFrame if no features found

    return feature_df, labels

# Load and preprocess the training and validation data
train_data_directory = 'F:\\MAIA\\third semester\\1. CAD\\Challenge two classes\CAD_Challenge1\\train\\train'
val_data_directory = 'F:\\MAIA\\third semester\\1. CAD\\Challenge two classes\\CAD_Challenge1\\val\\val\\val'

X_train, y_train = preprocess_and_extract_features(train_data_directory)
X_val, y_val = preprocess_and_extract_features(val_data_directory)

# Convert features and labels to NumPy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Apply feature scaling to your data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)

print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_val shape: {X_val_scaled.shape}")

# Create a linear SVC classifier
svc_classifier = SVC(kernel='linear')

# Train the SVC classifier
svc_classifier.fit(X_train_scaled, y_train)

# Predict on the validation set
y_pred_svc = svc_classifier.predict(X_val_scaled)

# Calculate accuracy
accuracy_svc = accuracy_score(y_val, y_pred_svc)
print(f"Validation Accuracy (SVC): {accuracy_svc}")