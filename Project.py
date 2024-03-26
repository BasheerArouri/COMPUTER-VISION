import os
from statistics import mode
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Threshold value
Thresholds = [18.0936, 12.967, 17.207, 16.269, 15.050, 17.678, 10.339, 7.871, 15.795, 12.156, 8.334, 9.455, 18.329,
              15.725, 14.544, 21.52, 14.459, 20.24, 19.057, 18.238, 21.127, 14.18, 10.7439, 20.013, 14.745, 9.0328,
              11.234, 21.466, 18.502, 15.802, 26.865, 19.865, 25.4581, 24.551, 23.087, 26.73, 18.544, 13.31, 24.32,
              18.591, 12.472, 14.081, 27.20, 24.0744, 22.557]

Thresholds_for_color_moments = [0.1195, 0.1297, 0.0460, 0.05903, 0.05793, 0.06391, 0.02972, 0.01918, 0.1005, 0.07627,
                                0.06914, 0.0562, 0.14882, 0.08347, 0.1222108]

Thresholds_for_additional_color_moments = [0.3858, 0.2656, 0.32207, 0.14159, 0.23960, 0.16554, 0.1389, 0.1403, 0.3161,
                                           0.4194, 0.4004, 0.2173, 0.3885, 0.3272, 0.20454]

general_counter = 0
overall_differences = {}
MAX_FOR_COLOR_HISTOGRAM = 0
MIN_FOR_COLOR_HISTOGRAM = 0


def clean_files():
    with open('Histograms_120.txt', 'w'):
        pass  # This will truncate the file to zero length

    with open('Histograms_180.txt', 'w'):
        pass  # This will truncate the file to zero length

    with open('Histograms_256.txt', 'w'):
        pass  # This will truncate the file to zero length

    with open('Histograms_for_R_channel', 'w'):
        pass  # This will truncate the file to zero length

    with open('Histograms_for_G_channel', 'w'):
        pass  # This will truncate the file to zero length

    with open('Histograms_for_B_channel', 'w'):
        pass  # This will truncate the file to zero length


# -----------------------------------------------------------------------------------------
# Truncate the files
clean_files()


# -----------------------------------------------------------------------------------------


def compute_histogram(image, pins):
    R, G, B = image.split()

    # Compute histograms for R, G, and B channels
    R_hist = cv2.calcHist([np.array(R)], [0], None, [pins], [0, 256])
    G_hist = cv2.calcHist([np.array(G)], [0], None, [pins], [0, 256])
    B_hist = cv2.calcHist([np.array(B)], [0], None, [pins], [0, 256])

    # R_hist = R_hist/np.sum(R_hist)
    # G_hist = G_hist/np.sum(G_hist)
    # B_hist = B_hist/np.sum(B_hist)

    # Calculate mean and standard deviation for each channel
    R_mean, R_std = np.mean(R_hist), np.std(R_hist)
    G_mean, G_std = np.mean(G_hist), np.std(G_hist)
    B_mean, B_std = np.mean(B_hist), np.std(B_hist)
    #
    # Z-score normalize histograms
    R_hist_normalized = (R_hist - R_mean) / R_std
    G_hist_normalized = (G_hist - G_mean) / G_std
    B_hist_normalized = (B_hist - B_mean) / B_std

    R_hist_normalized = [item for sublist in R_hist_normalized for item in sublist]
    G_hist_normalized = [item for sublist in G_hist_normalized for item in sublist]
    B_hist_normalized = [item for sublist in B_hist_normalized for item in sublist]

    # Concatenate the normalized histograms along the first axis to form a 3D histogram
    image_histogram_normalized = np.concatenate((R_hist_normalized, G_hist_normalized, B_hist_normalized))

    # Flatten the 3D histogram to a 1D array
    flattened_histogram = image_histogram_normalized.flatten()

    return flattened_histogram


def write_histograms_on_files():
    folder_path = 'dataset'
    pins = [120, 180, 256]

    for i in range(1000):
        file_name = '{}.jpg'.format(i)
        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path) and file_name.lower().endswith('.jpg'):
            with Image.open(full_file_path) as img:
                # Compute and flatten histogram for the image

                current_histogram_for_120 = compute_histogram(img, pins[0])
                current_histogram_for_180 = compute_histogram(img, pins[1])
                current_histogram_for_256 = compute_histogram(img, pins[2])

                # Convert NumPy array to a string separated by spaces
                current_histogram_str_120 = ' '.join(map(str, current_histogram_for_120))
                current_histogram_str_180 = ' '.join(map(str, current_histogram_for_180))
                current_histogram_str_256 = ' '.join(map(str, current_histogram_for_256))

                with open('Histograms_120.txt', 'a') as file:
                    # Append data to the file
                    file.write(current_histogram_str_120 + '\n')

                with open('Histograms_180.txt', 'a') as file:
                    # Append data to the file
                    file.write(current_histogram_str_180 + '\n')

                with open('Histograms_256.txt', 'a') as file:
                    # Append data to the file
                    file.write(current_histogram_str_256 + '\n')


def compute_difference_between_two_histograms_for_images(histogram_1, histogram_2):
    total_difference = sum((histogram_1 - histogram_2) ** 2)
    total_difference = np.sqrt(total_difference)
    return total_difference


def check_if_lower_than_the_threshold(total_difference, threshold):
    return total_difference < threshold


def rank_results(differences_with_this_testing_image):
    sorted_dict = dict(sorted(differences_with_this_testing_image.items(), key=lambda item: item[1]))
    return sorted_dict


def get_data_from_file(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    return content


def show_images(dictionary, pins):
    threshold = Thresholds[general_counter]
    # Create a list of 10 image paths (you can replace these with your image paths)
    image_paths = []

    folder_path = 'dataset'

    for key in dictionary.keys():
        if check_if_lower_than_the_threshold(dictionary[key], threshold):

            file_name = '{}.jpg'.format(key)
            full_file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(full_file_path) and file_name.lower().endswith('.jpg'):
                image_paths.append(full_file_path)

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))

    # Flatten the axes array to easily iterate over it
    axes = axes.ravel()

    # Loop over each image path and display it on a subplot
    # Add a main title to the entire figure
    fig.suptitle('12 Top Images, Pins = {}'.format(pins), fontsize=22, color='red')

    for i, image_path in enumerate(image_paths):

        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Convert from BGR to RGB (OpenCV reads images in BGR format, Matplotlib displays in RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image on the subplot
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis labels

        if i == 0:
            axes[0].set_title("Query Image (Image 1)")
        else:
            axes[i].set_title(f'Image {i + 1}')  # Set subplot title

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


def compute_accuracy_recall_precision_F1score(true_labels, predicted_labels):
    # Calculate accuracy
    TP = FP = TN = FN = 0
    for index in range(1000):

        if true_labels[index] == 1 and predicted_labels[index] == 1:
            TP = TP + 1
        elif true_labels[index] == 0 and predicted_labels[index] == 0:
            TN = TN + 1
        elif true_labels[index] == 0 and predicted_labels[index] == 1:
            FP = FP + 1
        else:
            FN = FN + 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = (2 * precision * recall) / (precision + recall)
    return accuracy, recall, precision, f1_score


def generate_true_labels(number):
    start = (number // 100) * 100  # Starting number for the current range
    binary_list = [0] * 1000  # Initialize a list of 1000 elements with zeros
    for i in range(start, start + 100):
        binary_list[i] = 1

    return binary_list


def display_accuracy_recall_precision_F1score(accuracy, recall, precision, F1score):
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {F1score * 100:.2f}%")


def test_query_images(training_histograms, pins):
    # Compute the histograms of the query images
    global overall_differences
    global general_counter
    global MAX_FOR_COLOR_HISTOGRAM
    global MIN_FOR_COLOR_HISTOGRAM

    # Because for each pins = [120, 180, 256]
    overall_differences = {}
    MAX_FOR_COLOR_HISTOGRAM = 0
    MIN_FOR_COLOR_HISTOGRAM = 0

    folder_path = 'Testing_images'

    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter out only image files based on the extension
    image_files = [f for f in all_files if f.lower().endswith('.jpg')]

    # Iterate over each image file without sorting
    for image_name in image_files:

        folder_path = 'Testing_images'
        current_image = 0
        differences_with_this_image = {}
        file_name = image_name
        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path) and file_name.lower().endswith('.jpg'):
            with Image.open(full_file_path) as img:

                # Compute and flatten histogram for the image
                histogram_for_current_testing_image = compute_histogram(img, pins)

                for current_histogram_for_training in training_histograms:
                    current_histogram_for_training = [float(x) for x in current_histogram_for_training.split(" ")]

                    # Calculate the difference between current training and query images
                    total_difference = compute_difference_between_two_histograms_for_images \
                        (current_histogram_for_training, histogram_for_current_testing_image)

                    if MAX_FOR_COLOR_HISTOGRAM < total_difference:
                        MAX_FOR_COLOR_HISTOGRAM = total_difference

                    if MIN_FOR_COLOR_HISTOGRAM > total_difference:
                        MIN_FOR_COLOR_HISTOGRAM = total_difference

                    differences_with_this_image[current_image] = total_difference
                    current_image = current_image + 1

        testing_idx = int(image_name.split(".")[0])
        images_after_ranking = rank_results(differences_with_this_image)

        # Show relevant images
        show_images(images_after_ranking, pins)

        # Save the differences for this testing image
        overall_differences[testing_idx] = differences_with_this_image

        # print_satisfied_images(images_after_ranking, pins)
        general_counter = general_counter + 1


# Find and plot the ROC curve
# I took 50 thresholds for the epochs

def compute_predictions(threshold, testing_image, differences):
    TP = FP = TN = FN = 0
    predicted_labels = [0 for _ in range(1000)]
    counter = 0

    for difference in differences:
        if check_if_lower_than_the_threshold(difference, threshold):
            predicted_labels[counter] = 1
        counter = counter + 1

    true_labels = generate_true_labels(testing_image)

    for i in range(len(true_labels)):
        if predicted_labels[i] == 1 and true_labels[i] == 1:
            TP = TP + 1
        elif predicted_labels[i] == 0 and true_labels[i] == 1:
            FN = FN + 1
        elif predicted_labels[i] == 1 and true_labels[i] == 0:
            FP = FP + 1
        else:
            TN = TN + 1

    return TP, FP, FN, TN


def get_TPR_FPR(MAX, MIN, differences, lista):
    number_of_testing_images = 15
    averages_TPR = []
    averages_FPR = []

    num_thresholds = 50
    interval_size = (MAX - MIN) / (num_thresholds - 1)

    thresholds_for_ROC_curve = [MIN + i * interval_size for i in range(num_thresholds)]

    # Computer the ROC curve for pins = 120, 180 and 256

    TPR_for_the_15_queries = 0
    FPR_for_the_15_queries = 0

    for threshold in thresholds_for_ROC_curve:
        for testing_image in lista:
            TP, FP, FN, TN = compute_predictions(threshold, testing_image,
                                                 differences[testing_image].values())
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            TPR_for_the_15_queries = TPR_for_the_15_queries + TPR
            FPR_for_the_15_queries = FPR_for_the_15_queries + FPR

        # Compute the average for 15 queries
        TPR_for_the_15_queries = TPR_for_the_15_queries / number_of_testing_images
        FPR_for_the_15_queries = FPR_for_the_15_queries / number_of_testing_images

        averages_TPR.append(TPR_for_the_15_queries)
        averages_FPR.append(FPR_for_the_15_queries)

    min_value = min(averages_TPR)
    max_value = max(averages_TPR)

    # Normalize the list to range between 0 and 1
    averages_TPR = [(x - min_value) / (max_value - min_value) for x in averages_TPR]

    min_value = min(averages_FPR)
    max_value = max(averages_FPR)

    # Normalize the list to range between 0 and 1
    averages_FPR = [(x - min_value) / (max_value - min_value) for x in averages_FPR]

    return averages_TPR, averages_FPR


def plot_ROC_curve(TPR_, FPR_):
    # Ensure that the lengths of TPR and FPR are the same
    if len(TPR_) != len(FPR_):
        raise ValueError("Lengths of TPR and FPR lists must be the same.")

    # Calculate AUC using the trapezoidal rule
    auc = np.trapz(TPR_, FPR_)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPR_, TPR_, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')  # Diagonal line for random guessing
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Set the limits of x and y axes to ensure '1' is on the border
    plt.ylim([0, 1])
    plt.show()


# --------------------------Color Histogram Testing Part--------------------------
# Test part

# List all files in the folder
all_files = os.listdir('Testing_images')

# Filter out only image files based on the extension
image_files = [f for f in all_files if f.lower().endswith('.jpg')]
list_of_testing_images = [int(x.split(".")[0]) for x in image_files]

write_histograms_on_files()

# Read the histograms from the files
Histograms_for_120_pins = get_data_from_file(filename='Histograms_120.txt')
Histograms_for_180_pins = get_data_from_file(filename='Histograms_180.txt')
Histograms_for_256_pins = get_data_from_file(filename='Histograms_256.txt')

# Testing the images for different pins
avg_precision = avg_recall = avg_accuracy = avg_F1_score = 0.0


def display_avg_metrics(pins):
    global avg_precision, avg_recall, avg_accuracy, avg_F1_score
    print("\nAverage values of metrics for 15 query images, with {} pins:".format(pins))
    avg_precision /= 15
    avg_recall /= 15
    avg_accuracy /= 15
    avg_F1_score /= 15

    display_accuracy_recall_precision_F1score(avg_accuracy, avg_recall, avg_precision, avg_F1_score)


def get_predicted_labels(testing_image, threshold, differences_values):
    predicted_output = [0] * 1000
    counter = 0
    differences = differences_values[testing_image].values()

    for difference in differences:
        if check_if_lower_than_the_threshold(difference, threshold):
            predicted_output[counter] = 1
        counter += 1

    return predicted_output


def evaluate_metrics(threshold, testing_image, differences_values):
    global avg_precision, avg_recall, avg_accuracy, avg_F1_score

    true_labels = generate_true_labels(testing_image)
    predicted_labels = get_predicted_labels(testing_image, threshold, differences_values)
    accuracy, recall, precision, F1score = compute_accuracy_recall_precision_F1score(true_labels, predicted_labels)

    avg_precision += precision
    avg_recall += recall
    avg_accuracy += accuracy
    avg_F1_score += F1score


# Part 120 pins --------------------------------------------------------------
test_query_images(Histograms_for_120_pins, 120)
TPR_, FPR_ = get_TPR_FPR(MAX_FOR_COLOR_HISTOGRAM, MIN_FOR_COLOR_HISTOGRAM, overall_differences, list_of_testing_images)
plot_ROC_curve(TPR_, FPR_)

for threshold, testing_image in zip(Thresholds[0:15], list_of_testing_images):
    evaluate_metrics(threshold, testing_image, overall_differences)

display_avg_metrics(pins=120)
print("------------------------------------------------------------------------------------")

# Part 180 pins --------------------------------------------------------------
avg_precision = avg_recall = avg_accuracy = avg_F1_score = 0.0
test_query_images(Histograms_for_180_pins, 180)
TPR_, FPR_ = get_TPR_FPR(MAX_FOR_COLOR_HISTOGRAM, MIN_FOR_COLOR_HISTOGRAM, overall_differences, list_of_testing_images)
plot_ROC_curve(TPR_, FPR_)

for threshold, testing_image in zip(Thresholds[15:30], list_of_testing_images):
    evaluate_metrics(threshold, testing_image, overall_differences)

display_avg_metrics(pins=180)
print("-------------------------------------------------------------------------------------")


# Part 256 pins --------------------------------------------------------------
avg_precision = avg_recall = avg_accuracy = avg_F1_score = 0.0
test_query_images(Histograms_for_256_pins, 256)
TPR_, FPR_ = get_TPR_FPR(MAX_FOR_COLOR_HISTOGRAM, MIN_FOR_COLOR_HISTOGRAM, overall_differences, list_of_testing_images)
plot_ROC_curve(TPR_, FPR_)

for threshold, testing_image in zip(Thresholds[30:45], list_of_testing_images):
    evaluate_metrics(threshold, testing_image, overall_differences)

display_avg_metrics(pins=256)


# ----------------------------------------------------------------------------------
# --------------------------Color Moments Testing Part--------------------------
MAX_FOR_COLOR_MOMENTS = 0
MIN_FOR_COLOR_MOMENTS = 0

MAX_FOR_ADDITION_COLOR_MOMENTS = 0
MIN_FOR_ADDITION_COLOR_MOMENTS = 0


def compute_histogram_for_color_moments(image, pins):
    R, G, B = image.split()

    # Compute histograms for R, G, and B channels
    R_hist = cv2.calcHist([np.array(R)], [0], None, [pins], [0, 256])
    G_hist = cv2.calcHist([np.array(G)], [0], None, [pins], [0, 256])
    B_hist = cv2.calcHist([np.array(B)], [0], None, [pins], [0, 256])

    R_hist = R_hist / np.sum(R_hist)
    G_hist = G_hist / np.sum(G_hist)
    B_hist = B_hist / np.sum(B_hist)

    R_hist_normalized = [item for sublist in R_hist for item in sublist]
    G_hist_normalized = [item for sublist in G_hist for item in sublist]
    B_hist_normalized = [item for sublist in B_hist for item in sublist]

    return R_hist_normalized, G_hist_normalized, B_hist_normalized


def write_histograms_R_G_B_in_file():
    folder_path = 'dataset'

    for i in range(1000):
        file_name = '{}.jpg'.format(i)
        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path) and file_name.lower().endswith('.jpg'):
            with Image.open(full_file_path) as img:
                # Compute and flatten histogram for the image

                current_R_histogram, current_G_histogram, current_B_histogram = \
                    compute_histogram_for_color_moments(img, 120)

                # Convert NumPy array to a string separated by spaces
                current_R_histogram = ' '.join(map(str, current_R_histogram))
                current_G_histogram = ' '.join(map(str, current_G_histogram))
                current_B_histogram = ' '.join(map(str, current_B_histogram))

                with open('Histograms_for_R_channel', 'a') as file:
                    # Append data to the file
                    file.write(current_R_histogram + '\n')

                with open('Histograms_for_G_channel', 'a') as file:
                    # Append data to the file
                    file.write(current_G_histogram + '\n')

                with open('Histograms_for_B_channel', 'a') as file:
                    # Append data to the file
                    file.write(current_B_histogram + '\n')


write_histograms_R_G_B_in_file()
Histograms_for_R_channel = get_data_from_file('Histograms_for_R_channel')
Histograms_for_G_channel = get_data_from_file('Histograms_for_G_channel')
Histograms_for_B_channel = get_data_from_file('Histograms_for_B_channel')


def compute_difference_moments(histograms_set1, histograms_set2, weights):
    total_difference = 0.0

    for channel in range(3):
        current_histogram_for_image1 = histograms_set1[channel]
        current_histogram_for_image2 = histograms_set2[channel]

        mean_channel_1 = np.mean(current_histogram_for_image1)
        mean_channel_2 = np.mean(current_histogram_for_image2)

        std_channel_1 = np.std(current_histogram_for_image1)
        std_channel_2 = np.std(current_histogram_for_image2)

        skew_channel_1 = np.mean((current_histogram_for_image1 - mean_channel_1) ** 3) \
                         / (std_channel_1 ** 3)
        skew_channel_2 = np.mean((current_histogram_for_image2 - mean_channel_2) ** 3) \
                         / (std_channel_2 ** 3)

        # Calculate the squared differences for each channel moment
        squared_diff_mean = (mean_channel_1 - mean_channel_2) ** 2
        squared_diff_std = (std_channel_1 - std_channel_2) ** 2
        squared_diff_skew = (skew_channel_1 - skew_channel_2) ** 2

        # Scale the squared differences using max values
        max_mean_diff = max(abs(mean_channel_1), abs(mean_channel_2))
        max_std_diff = max(std_channel_1, std_channel_2)
        max_skew_diff = max(abs(skew_channel_1), abs(skew_channel_2))

        scaled_diff_mean = squared_diff_mean / max_mean_diff ** 2
        scaled_diff_std = squared_diff_std / max_std_diff ** 2
        scaled_diff_skew = squared_diff_skew / max_skew_diff ** 2

        # Apply weights to the scaled differences
        total_difference += weights[0] * scaled_diff_mean + \
                            weights[1] * scaled_diff_std + \
                            weights[2] * scaled_diff_skew

    return total_difference


def apply_color_moments(testing_histogram, weights):
    global MAX_FOR_COLOR_MOMENTS, MIN_FOR_COLOR_MOMENTS

    rk = {}

    for index in range(len(Histograms_for_R_channel)):
        R_training_histogram = [float(x) for x in Histograms_for_R_channel[index].split(" ")]
        G_training_histogram = [float(x) for x in Histograms_for_G_channel[index].split(" ")]
        B_training_histogram = [float(x) for x in Histograms_for_B_channel[index].split(" ")]

        training_histogram = [R_training_histogram, G_training_histogram, B_training_histogram]
        difference = compute_difference_moments(training_histogram, testing_histogram, weights)

        if MAX_FOR_COLOR_MOMENTS < difference:
            MAX_FOR_COLOR_MOMENTS = difference

        if MIN_FOR_COLOR_MOMENTS > difference:
            MIN_FOR_COLOR_MOMENTS = difference

        rk[index] = difference
    return rk


def compute_difference_additional_moments(histograms_set1, histograms_set2, weights):
    total_difference = 0.0

    for channel in range(3):

        hist1, hist2 = histograms_set1[channel], histograms_set2[channel]

        mode_result_hist1 = mode(hist1)
        mode_result_hist2 = mode(hist2)

        # Add mode to the moments list
        moments = [
            (np.mean(hist1), np.mean(hist2)),  # mean
            (np.std(hist1), np.std(hist2)),  # std
            (
                np.mean((hist1 - np.mean(hist1)) ** 3) / (np.std(hist1) ** 3),
                np.mean((hist2 - np.mean(hist2)) ** 3) / (np.std(hist2) ** 3),
            ),  # skew
            (np.median(hist1), np.median(hist2)),  # median
            (
                np.mean((hist1 - np.mean(hist1)) ** 4) / (np.std(hist1) ** 4),
                np.mean((hist2 - np.mean(hist2)) ** 4) / (np.std(hist2) ** 4),
            ),  # kurtosis
            (mode_result_hist1, mode_result_hist2)  # mode
        ]

        scaled_diffs = []
        for i, (m1, m2) in enumerate(moments):
            squared_diff = (m1 - m2) ** 2

            # Calculate the max diff for scaling
            max_diff = max(abs(m1), abs(m2)) if i != 2 and i != 4 else max(abs(m1), abs(m2))

            if max_diff == 0:
                scaled_diffs.append(0)
            else:
                scaled_diffs.append(squared_diff / max_diff ** 2)

        total_difference += sum(w * d for w, d in zip(weights[:6], scaled_diffs))

    return total_difference


def apply_additional_color_moments(testing_histogram, weights):
    global MAX_FOR_ADDITION_COLOR_MOMENTS, MIN_FOR_ADDITION_COLOR_MOMENTS

    rk = {}

    for index in range(len(Histograms_for_R_channel)):

        R_training_histogram = [float(x) for x in Histograms_for_R_channel[index].split(" ")]
        G_training_histogram = [float(x) for x in Histograms_for_G_channel[index].split(" ")]
        B_training_histogram = [float(x) for x in Histograms_for_B_channel[index].split(" ")]

        training_histogram = [R_training_histogram, G_training_histogram, B_training_histogram]
        difference = compute_difference_additional_moments(training_histogram, testing_histogram, weights)

        if MAX_FOR_ADDITION_COLOR_MOMENTS < difference:
            MAX_FOR_ADDITION_COLOR_MOMENTS = difference

        if MIN_FOR_ADDITION_COLOR_MOMENTS > difference:
            MIN_FOR_ADDITION_COLOR_MOMENTS = difference

        rk[index] = difference

    return rk


def show_moments_images(dictionary, pins, counter_for_moments, thds):

    threshold = thds[counter_for_moments]
    # Create a list of 10 image paths (you can replace these with your image paths)
    image_paths = []

    folder_path = 'dataset'

    for key in dictionary.keys():
        if check_if_lower_than_the_threshold(dictionary[key], threshold):

            file_name = '{}.jpg'.format(key)
            full_file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(full_file_path) and file_name.lower().endswith('.jpg'):
                image_paths.append(full_file_path)

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))

    # Flatten the axes array to easily iterate over it
    axes = axes.ravel()

    # Loop over each image path and display it on a subplot
    # Add a main title to the entire figure
    fig.suptitle('12 Top Images, Pins = {}'.format(pins), fontsize=22, color='red')

    for i, image_path in enumerate(image_paths):

        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Convert from BGR to RGB (OpenCV reads images in BGR format, Matplotlib displays in RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image on the subplot
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis labels

        if i == 0:
            axes[0].set_title("Query Image (Image 1)")
        else:
            axes[i].set_title(f'Image {i + 1}')  # Set subplot title

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


# Moments weights (equals)
# Mean, std and skewness
W = [0.1, 0.7, 0.2]
# W = [0.33, 0.33, 0.34]
W_additional_moments = [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]
# W = [0.1, 0.3, 0.6]
folder_path = 'Testing_images'
ranked_differences = {}
differences_for_additional_moments = {}

# Iterate all over the testing images
all_files = os.listdir(folder_path)

# Filter out only image files based on the extension
image_files = [f for f in all_files if f.lower().endswith('.jpg')]

# Iterate over each image file without sorting
counter_for_moments = 0

# Iterate all over the testing images to find the equal weights and additional weights
for image_name in image_files:

    file_name = image_name
    full_file_path = os.path.join(folder_path, file_name)

    testing_index = int(file_name.split(".")[0])

    if os.path.isfile(full_file_path) and file_name.lower().endswith('.jpg'):
        with Image.open(full_file_path) as img:
            R_testing, G_testing, B_testing = compute_histogram_for_color_moments(img, 120)
            testing_histogram = [R_testing, G_testing, B_testing]

            rk = apply_color_moments(testing_histogram, W)
            additional_moments = apply_additional_color_moments(testing_histogram, W_additional_moments)

            ranked_differences[testing_index] = rk
            differences_for_additional_moments[testing_index] = additional_moments

            # Show relevant images for this image
            # This is for the original moments
            show_moments_images(rank_results(ranked_differences[testing_index]), 120, counter_for_moments
                                , Thresholds_for_color_moments)

            # This is for the additional moments
            show_moments_images(rank_results(differences_for_additional_moments[testing_index]),
                                120, counter_for_moments, Thresholds_for_additional_color_moments)

    counter_for_moments += 1

# Display some benefit metrics

# This is for original moments
avg_precision = avg_recall = avg_accuracy = avg_F1_score = 0.0
TPR_, FPR_ = get_TPR_FPR(MAX_FOR_COLOR_MOMENTS, MIN_FOR_COLOR_MOMENTS, ranked_differences, list_of_testing_images)
plot_ROC_curve(TPR_, FPR_)

for threshold, testing_image in zip(Thresholds_for_color_moments, list_of_testing_images):
    evaluate_metrics(threshold, testing_image, ranked_differences)

display_avg_metrics(pins=120)

# This is for additional moments
avg_precision = avg_recall = avg_accuracy = avg_F1_score = 0.0
TPR_, FPR_ = get_TPR_FPR(MAX_FOR_ADDITION_COLOR_MOMENTS, MIN_FOR_ADDITION_COLOR_MOMENTS,
                         differences_for_additional_moments, list_of_testing_images)
plot_ROC_curve(TPR_, FPR_)

for threshold, testing_image in zip(Thresholds_for_additional_color_moments, list_of_testing_images):
    evaluate_metrics(threshold, testing_image, differences_for_additional_moments)

display_avg_metrics(pins=120)
