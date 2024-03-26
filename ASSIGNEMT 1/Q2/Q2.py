import cv2
import numpy as np


# ----------------------------------First Part-----------------------------------
def insert_zeros(input_image, padded_by):
    image_with_zeros_above = np.vstack([np.zeros((padded_by, input_image.shape[1]), dtype=np.uint8), input_image])

    # Add rows of zeros (according to the filter size) below the last row
    image_with_zeros_below = np.vstack(
        [image_with_zeros_above, np.zeros((padded_by, input_image.shape[1]), dtype=np.uint8)])

    # Add columns of zeros (according to the filter size) to the left of column number 0
    image_with_zeros_left = np.hstack(
        [np.zeros((image_with_zeros_below.shape[0], padded_by), dtype=np.uint8), image_with_zeros_below])

    # Add columns of zeros (according to the filter size) to the right of column number 1
    image_with_zeros_both_sides = np.hstack(
        [image_with_zeros_left, np.zeros((image_with_zeros_left.shape[0], padded_by), dtype=np.uint8)])

    return image_with_zeros_both_sides


def generate_the_output_image(padded_by, rows, columns, filter, filter_size, image_with_zeros_both_sides, height,
                              width):
    output_image = np.zeros((height, width))
    for row in range(padded_by, rows - padded_by):
        for column in range(padded_by, columns - padded_by):

            kernel_image_for_this_pixel = np.zeros((filter_size, filter_size))
            # Do a filter for the current pixel
            for row_index in range(filter_size):
                current_row = image_with_zeros_both_sides[row - padded_by + row_index][
                              column - padded_by:column + padded_by + 1]
                kernel_image_for_this_pixel[row_index] = current_row

            kernel_image_for_this_pixel = kernel_image_for_this_pixel.flatten()
            # Do a dot product between the current filter for this pixel and the sobel x
            current_pixel = np.dot(kernel_image_for_this_pixel, filter)
            output_image[row - padded_by, column - padded_by] = current_pixel

    return output_image


# Define the function
def myImageFilter(input_image, filter):
    filter_size = int(np.sqrt(filter.size))
    padded_by = int((filter_size - 1) / 2)

    image_with_zeros_both_sides = insert_zeros(input_image, padded_by)

    rows = image_with_zeros_both_sides.shape[0]
    columns = image_with_zeros_both_sides.shape[1]

    height = input_image.shape[0]
    width = input_image.shape[1]

    output_image = generate_the_output_image(padded_by, rows, columns, filter, filter_size, image_with_zeros_both_sides,
                                             height, width)
    return output_image


def get_average_kernel(filter_size):
    return np.ones(filter_size, dtype=int)


def show_averaging_image(filter_size, image_path):
    filter = get_average_kernel(filter_size)
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    output_image = myImageFilter(input_image, filter)
    output_image = output_image / filter_size
    output_image = cv2.convertScaleAbs(output_image)
    cv2.imshow("Apply Kernel With Size = {}".format(filter_size), output_image)
    cv2.waitKey(0)


image_path_House1 = "House1.jpg"
image_path_House2 = "House2.jpg"

# Kernel = 3*3
show_averaging_image(9, image_path_House1)
show_averaging_image(9, image_path_House2)

# Kernel = 5*5
show_averaging_image(25, image_path_House1)
show_averaging_image(25, image_path_House2)


# ----------------------------------Second Part-----------------------------------

def get_gaussian_kernel(sigma):
    gaussian_kernel = cv2.getGaussianKernel(2 * sigma + 1, sigma)
    # Multiply the kernel by its transpose to get a 2D Gaussian filter
    return np.outer(gaussian_kernel, gaussian_kernel.transpose())


def show_gaussian_image(sigma, image_path):
    filter_size = (2 * sigma + 1) ** 2
    filter = get_gaussian_kernel(sigma)
    filter = filter.flatten()
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    output_image = myImageFilter(input_image, filter)

    # The Gaussian Filter is already did the normalization
    output_image = cv2.convertScaleAbs(output_image)
    cv2.imshow("Apply Gaussian Kernel With Size = {}".format(filter_size), output_image)
    cv2.waitKey(0)


# Show For sigma = 1
show_gaussian_image(1, image_path_House1)
show_gaussian_image(1, image_path_House2)
# Show For sigma = 2
show_gaussian_image(2, image_path_House1)
show_gaussian_image(2, image_path_House2)
# Show For sigma = 3
show_gaussian_image(3, image_path_House1)
show_gaussian_image(3, image_path_House2)


# ----------------------------------Third Part-----------------------------------

def show_sobel_image(image_path, sobel_x, sobel_y):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Gx = myImageFilter(input_image, sobel_x)
    Gy = myImageFilter(input_image, sobel_y)
    G = (Gx ** 2 + Gy ** 2) ** 0.5
    sobel_image = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('Normalized Gradient', cv2.convertScaleAbs(sobel_image))
    cv2.waitKey(0)


#  Test the sobel for the two images
sobel_x_kernel = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
sobel_y_kernel = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])
show_sobel_image(image_path_House1, sobel_x_kernel, sobel_y_kernel)
show_sobel_image(image_path_House2, sobel_x_kernel, sobel_y_kernel)


# ----------------------------------Forth Part-----------------------------------

def show_prewitt_image(image_path, prewitt_x, prewitt_y):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Gx = myImageFilter(input_image, prewitt_x)
    Gy = myImageFilter(input_image, prewitt_y)
    G = (Gx ** 2 + Gy ** 2) ** 0.5
    prewitt_image = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    output_image = cv2.convertScaleAbs(prewitt_image)
    cv2.imshow("Normalized Gradient: ", output_image)
    cv2.waitKey(0)


#  Test the prewitt for the two images
prewitt_x = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
prewitt_y = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

show_prewitt_image(image_path_House1, prewitt_x, prewitt_y)
show_prewitt_image(image_path_House2, prewitt_x, prewitt_y)
