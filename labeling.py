import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Define the directory containing segmented images
segmented_directory = 'CAD_Challenge1\\train\\train'  # Replace with the actual path

# Initialize lists to store features and labels
features = []
labels = []

# Function to extract features from an image (replace with your feature extraction method)
def extract_features_from_image(image):
    # Preprocess and extract features (replace this with your actual feature extraction code)
    # For example, you can use img = preprocess_and_extract_features(img)
    
    return img_features

# Process segmented images and assign labels
for filename in os.listdir(segmented_directory):
    image_path = os.path.join(segmented_directory, filename)
    img = cv2.imread(image_path)
    img_features = extract_features_from_image(img)

    # Determine the label based on the directory structure
    if "nevus" in image_path.lower():
        label = 1  # Positive label
    elif "others" in image_path.lower():
        label = 0  # Negative label
    else:
        continue  # Skip images in other folders

    # Append features and labels to their respective lists
    features.append(img_features)
    labels.append(label)

# Convert lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the combined array into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Now, X_train and y_train contain training data, and X_val and y_val contain validation data
def feature_scaling(train, val):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_val)
    return X_train, X_test

classifier = GaussianNB()
classifier.fit(X_train, y_train)

def metric(X_val, y_val):
    y_pred = classifier.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    accuracy_score(y_val, y_pred)
    return cm 