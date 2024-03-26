# Load the image
import cv2

image_path = "Q_4.jpg"


def apply_canny_detector(image_path, low_threshold, high_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_after_canny_detector = cv2.Canny(image, low_threshold, high_threshold)
    cv2.imshow("Image With Canny Detector:", image_after_canny_detector)
    cv2.waitKey(0)


# Test the Canny edge detector for three values of [Low threshold, High threshold]
apply_canny_detector(image_path, 5, 20)  # Low threshold
apply_canny_detector(image_path, 50, 90)  # Medium threshold
apply_canny_detector(image_path, 120, 220)  # High Threshold
