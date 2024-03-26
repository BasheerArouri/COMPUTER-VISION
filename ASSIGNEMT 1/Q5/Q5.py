# Loading walk_1.jpg and walk_2.jpg images
import cv2


def show_the_difference_between_two_images(image_1_path, image_2_path):
    image_1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)
    resulting_image_after_subtraction = image_1 - image_2

    for i in range(len(image_1)):
        for j in range(i):
            print("First = {}".format(image_1[i][j]))
            print("Second = {}".format(image_2[i][j]))
            print("Third = {}".format(resulting_image_after_subtraction[i][j]))
            print("\n\n\n")

    cv2.imshow("Image 1 - Image 2 =", resulting_image_after_subtraction)
    cv2.waitKey(0)


walk_1_path = "walk_1.jpg"
walk_2_path = "walk_2.jpg"

show_the_difference_between_two_images(walk_1_path, walk_2_path)
