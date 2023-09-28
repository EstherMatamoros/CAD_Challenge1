import cv2 as cv
import numpy as np

def main():
    img = cv.imread("nev00014.jpg")

    cv.imshow("Original image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    marker = cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41)))

    cv.imshow("Marker", marker)
    cv.waitKey(0)
    cv.destroyAllWindows()

    mask = img.copy()
    marker_prev = np.zeros_like(marker)
    marker_channels = [np.zeros_like(marker)] * 3
    marker_prev_channels = [np.zeros_like(marker)] * 3

    while True:
        marker_prev = marker.copy()
        cv.split(marker_prev, marker_prev_channels)
        marker = cv.dilate(marker, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        cv.min(marker, mask, marker)
        cv.split(marker, marker_channels)

        cv.imshow("Reconstruction in progress", marker)
        key = cv.waitKey(10)
        if key == 27:  # Press 'Esc' to exit the loop
            break

    cv.destroyAllWindows()
    cv.imshow("Reconstruction result", marker)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow("Stars result", img - marker)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
