import cv2
import numpy as np
from matplotlib import pyplot as plt

def perform_segmentation(hair_removed_image):
    # Convert to grayscale
    hair_removed_gray = cv2.cvtColor(hair_removed_image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(hair_removed_gray, (5, 5), 0)

    # Otsu's Thresholding
    _, thresholded_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert to binary image (0 and 255)
    binary_segmented_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)
    
    # Print unique values after thresholding
    unique_values = np.unique(binary_segmented_image)
 
    # Explicit additional processing steps (if needed)
    background_mask = np.zeros_like(binary_segmented_image)
    background_mask[:30, :] = 255
    binary_segmented_image_no_triangles = cv2.bitwise_and(binary_segmented_image, binary_segmented_image, mask=cv2.bitwise_not(background_mask))

    inverted_image = cv2.bitwise_not(binary_segmented_image_no_triangles)

    h, w = inverted_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted_image, mask, (0, 0), 0)

    kernel_opening = np.ones((5, 5), np.uint8)
    segmented_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    kernel_smoothing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    segmented_image_smoothed = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel_smoothing, iterations=1)
    segmented_image_smoothed = cv2.morphologyEx(segmented_image_smoothed, cv2.MORPH_CLOSE, kernel_smoothing, iterations=1)

    return segmented_image_smoothed

def get_roi_mask(segmented_image_smoothed):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a binary mask with the same dimensions as the segmented image
    h, w = segmented_image_smoothed.shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw rectangles around each detected contour
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(roi_mask, (x, y), (x + width, y + height), 255, -1)  # Fills the bounding box with white (255)

    return roi_mask

def extract_rois(original_image, roi_mask):
    # Check if the ROI mask has any non-zero pixels
    if np.any(roi_mask):
        # Multiply the original image with the ROI mask
        roi_image = cv2.bitwise_and(original_image, original_image, mask=roi_mask)
        return roi_image
    else:
        # No ROIs found, return the original image
        return original_image