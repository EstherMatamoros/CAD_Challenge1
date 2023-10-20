import cv2
import numpy as np

# Load the original image
original_image = cv2.imread('nev00014.jpg')

# Step 1: Apply Gaussian Filter for Noise Reduction
blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

# Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# Step 3: Thresholding for Segmentation
_, thresholded_image = cv2.threshold(clahe_image, 160, 255, cv2.THRESH_BINARY)

# Step 4: Opening and Closing Operations to Remove Hair (with linear structural element)
kernel = np.ones((7, 7), np.uint8)
opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Step 5: Find Contours
contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on a copy of the original image
contour_image = original_image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Opening and Closing', closed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 6: Apply Contours to Original Image and Extract Features
features_within_contours = []

for contour in contours:
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    features_within_contour = cv2.bitwise_and(clahe_image, clahe_image, mask=mask)
    
    # Extract features from the region within the contour
    # You can add your feature extraction code here for 'features_within_contour'
    
    features_within_contours.append(features_within_contour)

# 'features_within_contours' now contains the extracted features within each contour
print('Features within contours', features_within_contours)
