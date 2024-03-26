# For the first image ----> "Noisyimage1.jpg"
# Remove (Not Reduce) the salt and pepper noisy data by the median filter
import cv2

Noisyimage1_path = "Noisyimage1.jpg"
Noisyimage2_path = "Noisyimage2.jpg"


def show_median_image(image_path, filter_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_after_median_filter = cv2.medianBlur(image, filter_size)
    cv2.imshow("Image With Median Filter ----> 5*5 Kernel Size", image_after_median_filter)
    cv2.waitKey(0)


def show_averaging_image(image_path, filter_size):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_after_box_filter_with_solt_and_pepper = cv2.boxFilter(input_image, -1, filter_size)
    cv2.imshow("Image With Averaging Filter ----> 5*5 Kernel Size", image_after_box_filter_with_solt_and_pepper)
    cv2.waitKey(0)


# Apply 5 by 5 averaging filtration
filter_size = (5, 5)
show_averaging_image(Noisyimage1_path, filter_size)
show_averaging_image(Noisyimage2_path, filter_size)

# Apply 5 by 5 median filtration
show_median_image(Noisyimage1_path, filter_size[0])
show_median_image(Noisyimage2_path, filter_size[0])
