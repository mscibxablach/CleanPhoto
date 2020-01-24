import os
import cv2
import numpy as np
import math
import functools
import imutils
from collections import Counter
from sklearn.cluster import KMeans
from enum import Enum
from matplotlib import pyplot as plt
from ImageOperations import ImageOperations
from ImageProcessor import ImageProcessor
from MaskPosition import MaskPosition
from PlotBoundDetector import PlotBoundDetector

print(cv2.__version__)

def get_filename_from_path(path):
    return os.path.basename(path)

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 10:
                return True
            elif i == row1-1 and j == row2-1:
                return False

# Moved
def get_shape(image):
    height, width, _ = image.shape
    return height, width

# Moved and rename
def morph_image(image):
    image = image.copy()
    kernel = np.ones((5, 5), np.uint8)
    _, mask = cv2.threshold(image, 107, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# Moved
def get_ox_coordinates(image):
    image = image.copy()
    morphed_image = morph_image(image)
    edges = cv2.Canny(morphed_image, 100, 200)
    lines = cv2.HoughLines(edges, 1, math.pi / 180, 1)

    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # cv2.line(morphed_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

        return x1, x2, y1, y2


#         return morphed_image

# Moved
def mask_image(image, maskOrientation):
    copied_image = image.copy()
    morphed_image = morph_image(copied_image)
    x1, x2, y1, y2 = get_ox_coordinates(morphed_image)
    height, width = get_shape(morphed_image)

    if maskOrientation is MaskPosition.TOP:
        mask = np.zeros((height, width), np.uint8)
        mask = cv2.rectangle(mask, (x1, y1), (width, height), (255, 255, 255), cv2.FILLED)
        return cv2.bitwise_and(morphed_image, morphed_image, mask=mask)
    else:
        mask = np.zeros((height, width), np.uint8)
        mask = cv2.rectangle(mask, (x2, y2), (0, 0), (255, 255, 255), cv2.FILLED)
        return cv2.bitwise_and(morphed_image, morphed_image, mask=mask)

# Moved
def cut_image_on_ox_axis(image, maskPosition):
    copied_image = image.copy()
    morphed_image = morph_image(copied_image)
    x1, x2, y1, y2 = get_ox_coordinates(morphed_image)
    height, width = get_shape(morphed_image)

    if maskPosition is MaskPosition.BOTTOM:
        return copied_image[y1:height, x1:width]
    if maskPosition is MaskPosition.TOP:
        return copied_image[0:y2, 0:x2]

# Moved
def read_image(filename):
    return cv2.imread(filename)

# Moved
def manual_threshold(image):
    image_copied = image.copy()
    rows, cols = image_copied.shape

    for i in range(rows):
        for j in range(cols):
            pixel_value = image_copied[i, j]
            if pixel_value <= 95:
                image_copied[i, j] = 0

    return image_copied

# Moved
def get_continous_chunks(array):
    result = []
    min_value = min(array)
    i = 0
    while i < len(array) - 1:
        print('row', i)
        value = array[i]
        if value > min_value:
            chunk = [i]
            counter = i + 1
            for j in range(counter, len(array)):
                print(j)
                value = array[j]
                if value > min_value:
                    chunk.append(j)
                else:
                    i = j
                    result.append(chunk)
                    break
            else:
                result.append(chunk)
                i = j
                continue
        elif value <= min_value:
            i += 1
            continue
    return result

# Moved
def get_histogram(image, title):
    result = []
    image_copy = image.copy()
    rows, cols = image.shape

    for j in range(cols):
        sum = 0
        for i in range(rows):
            sum = sum + image[i, j]
        result.append(sum)

    plt.plot(result)
    plt.title(title)
    plt.show()

    return result


def get_all_images(path):
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(os.path.join(path,name))
    return files

# Moved
def resize_image(image):
    width, height, _ = image.shape
    return image[20:width-50, 0:height-50]

# Moved
def convert_to_gray(image):
    image = image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Moved
def draw_chunks(chunks, image):
    width, height, _ = image.shape

    for chunk in chunks:
        color = list(map(lambda x: int(x), list(np.random.choice(range(256), size=3))))
        start_column = chunk[0]
        end_column = chunk[-1]
        cv2.line(image, (start_column, 0), (start_column, height), color, 1)
        cv2.line(image, (end_column, 0), (end_column, height), color, 1)

# Moved
def get_avrage_length(chunks):
    sum_len = sum(len(c) for c in chunks)
    chunks_count = len(chunks)
    return round(sum_len /chunks_count)

# Moved
def filter_chunks(chunks):
    avg_len = get_avrage_length(chunks)
    result = list(filter(lambda x: len(x) >= avg_len, chunks))
    return result

# Moved
def get_plot_bound(image):
    image = image.copy()
    gray = convert_to_gray(image)

    manual_thresholded = manual_threshold(gray)
    bilateral_filtered = cv2.bilateralFilter(manual_thresholded.copy(), 11, 75, 75)
    _, thresholded = cv2.threshold(bilateral_filtered, 70, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    histogram = get_histogram(dilated, '')

    continous_chunks = get_continous_chunks(histogram)
    continous_chunks = filter_chunks(continous_chunks)

    draw_chunks(continous_chunks, image)

    return image

# Moved
def remove_ox(image):
    image = image.copy()
    x1, x2, y1, y2 = get_ox_coordinates(image)
    cv2.line(image, (x1, y1), (x2, y2), (0,0,0), 5)

    return image

# Moved
def delete_text(image):
    image = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Every color except white
    lower = np.array([0, 42, 0])
    higher = np.array([179, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower, higher)

    image[mask > 0] = (255, 255, 255)

    return cv2.bitwise_not(image, image, mask=mask)


def main_v2():
    output_path = 'Result'
    image_paths = get_all_images('Photos')
    # for image_path in image_paths:
    #     image = read_image(image_path)
    #     image = delete_text(image)
    #     image = remove_ox(image)
    #     result = get_plot_edges(image)
    #     filename = get_filename_from_path(image_path)
    #     path = os.path.join(output_path, filename)
    #     cv2.imwrite(path, result)
    image_processor = ImageProcessor()
    plot_bound_detector = PlotBoundDetector()
    for image_path in image_paths:
        print(image_path)
        image = ImageOperations.read_image(image_path)
        image = ImageOperations.resize_image(image)
        image = image_processor.delete_text(image)
        
        bottom_part = image_processor.cut_image_on_ox_axis(image, MaskPosition.BOTTOM)
        top_part = image_processor.cut_image_on_ox_axis(image, MaskPosition.TOP)

        bottom_part = image_processor.remove_ox(bottom_part)
        top_part = image_processor.remove_ox(top_part)

        bottom_result = plot_bound_detector.get_plot_bound(bottom_part)
        top_result = plot_bound_detector.get_plot_bound(top_part)

        result = np.concatenate((top_result, bottom_result), axis=0)
        cv2.imwrite(os.path.join(output_path, get_filename_from_path(image_path)), result)


def main():
    main_v2()


if __name__ == '__main__':
    main()










