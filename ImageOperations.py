import cv2
import numpy as np


class ImageOperations:
    @staticmethod
    def convert_to_gray(image):
        image = image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def get_shape(image):
        if len(image.shape) > 2:
            height, width, _ = image.shape
            return height, width
        else:
            height, width = image.shape
            return height, width

    @staticmethod
    def morph_open_erode_close_image(image):
        image = image.copy()
        kernel = np.ones((5, 5), np.uint8)
        _, mask = cv2.threshold(image, 107, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.erode(mask, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    @staticmethod
    def read_image(filename):
        return cv2.imread(filename)

    @staticmethod
    def color_threshold(image, threshold = 95):
        image_copied = image.copy()
        rows, cols = image_copied.shape

        for i in range(rows):
            for j in range(cols):
                pixel_value = image_copied[i, j]
                if pixel_value <= threshold:
                    image_copied[i, j] = 0

        return image_copied

    @staticmethod
    def resize_image(image):
        width, height, _ = image.shape
        return image[20:width - 50, 0:height - 50]

    @staticmethod
    def convert_to_gray(image):
        image = image.copy()

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
