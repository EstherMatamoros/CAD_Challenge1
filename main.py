import os
import cv2
from preprocessing.preprocessing import contrast_stretching, remove_hair
from segmentation.segmentation import perform_segmentation
from data_loader.data_loading import load_images_from_folder
from tqdm import tqdm  # Add this line for progress bar
import numpy as np
from matplotlib import pyplot as plt

def main():
    train_folder = 'C:/Users/Administrador/Documents/0. MAIA/3. Spain/1.CAD/1.Project_SEP/Skinbinary_train/nevus/'
    output_folder = 'C:/Users/Administrador/Documents/processed_images/'  # Specify your output folder
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    images = load_images_from_folder(train_folder)

    for img, filename in tqdm(images, desc="Processing images", unit="image"):
        # Convert to grayscale
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check if the grayscale image is empty
        if image_gray.size == 0:
            print("Error: Grayscale image is empty.")
            continue

        # Resize the image to 600x600
        image_resized = cv2.resize(image_gray, (600, 600))

        # Noise removal with a 3x3 median filter
        image_noise_removed = cv2.medianBlur(image_resized, 3)

        # Contrast stretching on the grayscale image
        image_contrast_stretched = contrast_stretching(image_noise_removed)

        # Hair removal using bottom-hat filtering
        hair_removed_image = remove_hair(image_contrast_stretched)

        # Convert enhanced image to gray only if it's not already grayscale
        if len(image_resized.shape) == 3:  # Check if the image has 3 color channels (not grayscale)
            enhanced_gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_gray_image = image_resized  # If it's already grayscale, no need to convert

        '''# Print image type information
        if len(enhanced_gray_image.shape) == 2:
            print("Enhanced Gray Image is Grayscale")
        elif len(enhanced_gray_image.shape) == 3 and enhanced_gray_image.shape[2] == 3:
            print("Enhanced Gray Image is RGB")
        elif len(enhanced_gray_image.shape) == 3 and enhanced_gray_image.shape[2] == 1:
            print("Enhanced Gray Image is Binary")'''

        '''# Print statistics for Enhanced Gray Image
        print_image_statistics(enhanced_gray_image, 'Enhanced Gray')

        # Print statistics for Hair-Removed Image
        print_image_statistics(hair_removed_image, 'Hair-Removed')'''

        '''# Plotting the segmentation results
        plt.figure(figsize=(15, 6))

        # Enhanced gray image
        plt.subplot(1, 3, 1)
        plt.imshow(enhanced_gray_image, cmap='gray')
        plt.title('Enhanced Gray Image')
        plt.axis('off')

        # Hair removal result
        plt.subplot(1, 3, 2)
        plt.imshow(hair_removed_image)
        plt.title('Hair-Removed Image')
        plt.axis('off')'''

        # Perform segmentation
        segmented_image = perform_segmentation(hair_removed_image)

        '''plt.subplot(1, 3, 3)
        plt.imshow(segmented_image, cmap='gray')
        plt.title('Segmented Image')
        plt.axis('off')

        plt.show()'''

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, segmented_image)

if __name__ == "__main__":
    main()