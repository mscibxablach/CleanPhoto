import os
import cv2
import numpy as np
import math
import imutils
from collections import Counter
from sklearn.cluster import KMeans
from enum import Enum
from matplotlib import pyplot as plt

print(cv2.__version__)


class MaskPosition(Enum):
    BOTTOM = 1
    TOP = 2


def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 10:
                return True
            elif i == row1-1 and j == row2-1:
                return False


def get_shape(image):
    height, width = image.shape
    return height, width


def morph_image(image):
    image = image.copy()
    kernel = np.ones((5, 5), np.uint8)
    _, mask = cv2.threshold(image, 107, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


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


def cut_image_on_ox_axis(image, maskPosition):
    copied_image = image.copy()
    morphed_image = morph_image(copied_image)
    x1, x2, y1, y2 = get_ox_coordinates(morphed_image)
    height, width = get_shape(morphed_image)

    if maskPosition is MaskPosition.BOTTOM:
        return copied_image[y1:height, x1:width]
    if maskPosition is MaskPosition.TOP:
        return copied_image[0:y2, 0:x2]


def read_image(filename):
    return cv2.imread(filename)


def manual_threshold(image):
    image_copied = image.copy()
    rows, cols = image_copied.shape

    for i in range(rows):
        for j in range(cols):
            pixel_value = image_copied[i, j]
            if pixel_value <= 95:
                image_copied[i, j] = 0

    return image_copied


def get_continous_chunks(array):
    result = []
    min_value = min(array)
    i = 0
    while i < len(array):
        value = array[i]
        if value > min_value:
            chunk = [i]
            counter = i + 1
            for j in range(counter, len(array)):
                value = array[j]
                if value > min_value:
                    chunk.append(j)
                else:
                    i = j
                    result.append(chunk)
                    break
            else:
                continue
        elif value <= min_value:
            i += 1
            continue
    return result


def get_histogram(image):
    result = []
    image_copy = image.copy()
    rows, cols = image.shape

    for j in range(cols):
        sum = 0
        for i in range(rows):
            sum = sum + image[i, j]
        result.append(sum)
    plt.plot(result)
    plt.show()

    return result


def main():
    image_path = 'Photos/43.jpg'
    image = read_image(image_path)
    copied_image = image.copy()
    cv2.imshow('original image', copied_image)
    copied_image = cv2.cvtColor(copied_image, cv2.COLOR_BGR2GRAY)
    width, height = copied_image.shape
    copied_image = copied_image[20:width-50, 0:height-50]
    bottom_part = cut_image_on_ox_axis(copied_image, MaskPosition.BOTTOM)
    top_part = cut_image_on_ox_axis(copied_image, MaskPosition.TOP)

    manaul_threshold_bottom = manual_threshold(bottom_part)
    manual_threshold_top = manual_threshold(top_part)


    bilateral_filter_bottom = cv2.bilateralFilter(manaul_threshold_bottom.copy(), 11, 75, 75)
    bilateral_filter_top = cv2.bilateralFilter(manual_threshold_top.copy(), 11, 75, 75)

    _, thresholded_bottom = cv2.threshold(manaul_threshold_bottom, 70, 255, cv2.THRESH_BINARY)
    _, thresholded_top = cv2.threshold(manual_threshold_top, 70, 255, cv2.THRESH_BINARY)

    # max_value_in_histogram = max(histogram)

    ## Dilate of manual threshold image
    # kernel = np.ones((3, 3), np.uint8)
    # dilated_bottom = cv2.dilate(manaul_threshold_bottom, kernel, iterations=3)
    # dilated_top = cv2.dilate(manual_threshold_top, kernel, iterations=3)

    kernel = np.ones((3, 3), np.uint8)
    # kernel = kernel.astype(np.uint8)
    # kernel = np.ones((5, 5), np.uint8)
    # print(kernel)
    # morph_open_bottom = cv2.morphologyEx(thresholded_bottom, cv2.MORPH_OPEN, kernel)
    # morph_close_bottom = cv2.morphologyEx(morph_open_bottom, cv2.MORPH_CLOSE, kernel)
    dilated_bottom = cv2.dilate(thresholded_bottom, kernel, iterations=2)

    kernel_closing = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], np.uint8)
    morph_close_bottom = cv2.morphologyEx(thresholded_bottom, cv2.MORPH_CLOSE, kernel_closing)

    canny_bottom = cv2.Canny(thresholded_bottom, 30, 200)
    canny_top = cv2.Canny(thresholded_top, 30, 200)

    bottom_result = np.concatenate((bottom_part, manaul_threshold_bottom, bilateral_filter_bottom, thresholded_bottom, dilated_bottom, morph_close_bottom, canny_bottom), axis=0)
    top_result = np.concatenate((top_part, manual_threshold_top, bilateral_filter_top, thresholded_top, canny_top), axis=0)

    histogram = get_histogram(dilated_bottom)
    print(histogram)
    print(histogram[122])
    print(histogram[123])

    bottom_part = cv2.cvtColor(bottom_part, cv2.COLOR_GRAY2BGR)

    continous_chunks = get_continous_chunks(histogram)
    for chunk in continous_chunks:
        color = list(map(lambda x: int(x), list(np.random.choice(range(256), size=3))))
        chunk_len = len(chunk)
        start_column = chunk[0]
        end_column = chunk[-1]
        cv2.line(bottom_part, (start_column, 0), (start_column, height), color, 1)
        cv2.line(bottom_part, (end_column, 0), (end_column, height), color, 1)

    # contours_bottom, _ = cv2.findContours(canny_bottom.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    # cv2.drawContours(bottom_part, contours_bottom, -1, [0, 255, 0], 1)

    # contours_and_area = list(map(lambda x: (x, cv2.contourArea(x)), contours_bottom))
    # print(contours_and_area)
    # for cnt in contours_bottom:
    #     color = list(map(lambda x: int(x), list(np.random.choice(range(256), size=3))))
    #     cv2.drawContours(bottom_part, [cnt], 0, color, 1)

    cv2.imshow('result', bottom_part)
    cv2.imshow('bottom', bottom_result)
    cv2.imshow('top', top_result)

    cv2.waitKeyEx()


if __name__ == '__main__':
    main()
