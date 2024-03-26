# Compute gradient magnitude for attached image “Q4_Image”
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------First Part-----------------------------------
def get_sobel_magnitude_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel filter
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalize the result to be in the range [0, 255]
    cv2.normalize(sobel_magnitude, sobel_magnitude, 0, 255, cv2.NORM_MINMAX)
    sobel_magnitude_stretched = cv2.convertScaleAbs(sobel_magnitude)
    return sobel_magnitude_stretched


def show_sobel_image(image):
    cv2.imshow("Sobel Image:", image)
    cv2.waitKey(0)


image_path = "Q_4.jpg"
# Show the stretched magnitude image
sobel_magnitude_stretched = get_sobel_magnitude_image(image_path)
show_sobel_image(sobel_magnitude_stretched)


# ----------------------------------Second Part-----------------------------------
def show_mag_histogram(gradient):
    gradient_histogram, bins = np.histogram(gradient.flatten(), bins=256)
    gradient_histogram = gradient_histogram / sum(gradient_histogram)
    # Plot the histogram
    plt.bar(bins[:-1], gradient_histogram, width=0.7, color='red')
    plt.title('Image Magnitude Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()


# Show the histogram of the magnitude
show_mag_histogram(sobel_magnitude_stretched)


# ----------------------------------Third Part-----------------------------------
def get_sobel_orientation_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel filter
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate the gradient Orientation
    sobel_orientation = np.arctan2(sobel_y, sobel_x)
    cv2.normalize(sobel_orientation, sobel_orientation, 0, 360, cv2.NORM_MINMAX)
    sobel_orientation_stretched = cv2.convertScaleAbs(sobel_orientation)
    return sobel_orientation_stretched


# Show the stretched orientation image
sobel_orientation_stretched = get_sobel_orientation_image(image_path)
show_sobel_image(sobel_orientation_stretched)


# ----------------------------------Forth Part-----------------------------------
def show_ori_histogram(gradient):
    gradient_histogram, bins = np.histogram(gradient.flatten(), bins=360)
    # Plot the histogram
    plt.bar(bins[:-1], gradient_histogram, width=0.7, color='blue')
    plt.title('Image Orientation Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()


# Show the histogram of the orientation
show_ori_histogram(sobel_orientation_stretched)
