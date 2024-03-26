import random
import cv2
import numpy as np

# ----------------------------------First Part----------------------------------
# Load the image
image_path = "image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("256*256 bits size and 8-bits grey level Image", image)
cv2.waitKey(0)
copy_image = np.copy(image)  # Copy the image for the last part

# ----------------------------------Second Part----------------------------------
# Apply the power-law transformation
c = 1
gamma = 0.4  # Define the tuning parameters for this equation s = cr^gamma -----> where c is constant, r is the
# input pixel and gamma is the gamma correction
image_after_normalized = image.astype(np.float32) / 255.0
image_after_power_low = c * np.power(image_after_normalized, gamma)  # s = c*r^gamma
image_after_power_low = (image_after_power_low * 255).astype(np.uint8)  # Convert the result to uint8 (
# standard image format) -----> The output image will be grayed, because we have gamma = 0.4 (less than 1 ---> Log)
cv2.imshow("Image With Power Low Transformation ----> gamma = 0.4", image_after_power_low)
cv2.waitKey(0)

# ----------------------------------Third Part-----------------------------------
mean = 0  # Mean = 0 for the Gaussian noise
variance = 40  # Variance for the gaussian filter, by default the mean = 0
gaussian_noise = np.random.normal(mean, np.sqrt(variance), image.shape)
image_after_gaussian_noise = image.astype(np.float32) + gaussian_noise
image_after_gaussian_noise = np.clip(image_after_gaussian_noise, 0, 255)
image_after_gaussian_noise = np.uint8(image_after_gaussian_noise)
cv2.imshow("Image With Gaussian Noise ----> Variance = 40 And Mean = 0", image_after_gaussian_noise)
cv2.waitKey(0)

# ----------------------------------Forth Part-----------------------------------
kernel_size = (5, 5)  # Kernel Size = 5 * 5
image_after_box_filter = cv2.boxFilter(image_after_gaussian_noise, -1, kernel_size, normalize=True)
cv2.imshow("Image With Box Filter ----> Kernel size = 5*5", image_after_box_filter)
cv2.waitKey(0)

# ----------------------------------Fifth Part-----------------------------------

# Adding salt and pepper noisy data
salt_probability_of_the_image = 0.1
pepper_probability_of_the_image = 0.1

number_of_salt_noises = int(salt_probability_of_the_image * image.size)
number_of_pepper_noises = int(pepper_probability_of_the_image * image.size)

salts_pixels = [[]]
pepper_pixels = [[]]

width = image.shape[0]
height = image.shape[1]

for i in range(0, number_of_salt_noises):
    salt_pixel_x = random.randint(0, width-1)
    salt_pixel_y = random.randint(0, height-1)

    pepper_pixel_x = random.randint(0, width-1)
    pepper_pixel_y = random.randint(0, height-1)

    image[salt_pixel_x][salt_pixel_y] = 255
    image[pepper_pixel_x][pepper_pixel_y] = 0

cv2.imshow("Image With Salt and Pepper Noise", image)
cv2.waitKey(0)

# Remove (Not Reduce) the salt and pepper noisy data by the median filter
kernel_size = 7
image_after_median_filter = cv2.medianBlur(image, kernel_size)
cv2.imshow("Image With Median Filter ----> 7*7 Kernel Size", image_after_median_filter)
cv2.waitKey(0)

# ----------------------------------Sixth Part-----------------------------------
kernel_size = (7, 7)
image_after_box_filter_with_solt_and_pepper = cv2.boxFilter(image, -1, kernel_size)
cv2.imshow("Image With Box Filter And Solt And Pepper Noise ----> Kernel size = 7*7",
           image_after_box_filter_with_solt_and_pepper)
cv2.waitKey(0)

# ----------------------------------Seventh Part-----------------------------------

sobel_X_kernel = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
sobel_Y_kernel = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])

# Add one row of zeros above row number 0
image_with_zeros_above = np.vstack([np.zeros((1, copy_image.shape[1]), dtype=np.uint8), copy_image])

# Add one row of zeros below row number 1
image_with_zeros_below = np.vstack([image_with_zeros_above, np.zeros((1, copy_image.shape[1]), dtype=np.uint8)])

# Add one column of zeros to the left of column number 0
image_with_zeros_left = np.hstack(
    [np.zeros((image_with_zeros_below.shape[0], 1), dtype=np.uint8), image_with_zeros_below])

# Add one column of zeros to the right of column number 1
image_with_zeros_both_sides = np.hstack(
    [image_with_zeros_left, np.zeros((image_with_zeros_left.shape[0], 1), dtype=np.uint8)])

rows = image_with_zeros_both_sides.shape[0]
columns = image_with_zeros_both_sides.shape[1]

height = copy_image.shape[0]
width = copy_image.shape[1]

G = np.zeros((height, width), dtype=int)

for row in range(1, rows - 1):
    for column in range(1, columns - 1):
        # Do a filter for the current pixel
        first_row = image_with_zeros_both_sides[row - 1][column - 1:column + 2]
        second_row = image_with_zeros_both_sides[row][column - 1:column + 2]
        third_row = image_with_zeros_both_sides[row + 1][column - 1:column + 2]

        kernel_image_for_this_pixel = np.concatenate([first_row, second_row, third_row])

        # Do a dot product between the current filter for this pixel and the sobel x
        gx = np.dot(sobel_X_kernel, kernel_image_for_this_pixel)
        gy = np.dot(sobel_Y_kernel, kernel_image_for_this_pixel)

        G[row - 1, column - 1] = (gx ** 2 + gy ** 2) ** 0.5

#Normalize the Gradient
sobel_mag_image = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('Normalized Gradient', cv2.convertScaleAbs(sobel_mag_image))
cv2.waitKey(0)
