import numpy as np
import cv2
import math
from MaskPosition import MaskPosition
from ImageOperations import ImageOperations


class ImageProcessor:
    def mask_image(self, image, maskOrientation):
        copied_image = image.copy()
        morphed_image = ImageOperations.morph_open_erode_close_image(copied_image)
        x1, x2, y1, y2 = self.get_ox_coordinates(morphed_image)
        height, width = ImageProcessor.get_shape(morphed_image)

        if maskOrientation is MaskPosition.TOP:
            mask = np.zeros((height, width), np.uint8)
            mask = cv2.rectangle(mask, (x1, y1), (width, height), (255, 255, 255), cv2.FILLED)
            return cv2.bitwise_and(morphed_image, morphed_image, mask=mask)
        else:
            mask = np.zeros((height, width), np.uint8)
            mask = cv2.rectangle(mask, (x2, y2), (0, 0), (255, 255, 255), cv2.FILLED)
            return cv2.bitwise_and(morphed_image, morphed_image, mask=mask)

    def split_image(self, image):
        copied_image = image.copy()
        bottom = self.cut_image_on_ox_axis(copied_image, MaskPosition.BOTTOM)
        top = self.cut_image_on_ox_axis(copied_image, MaskPosition.TOP)

        return bottom, top

    def cut_image_on_ox_axis(self, image, maskPosition):
        copied_image = image.copy()
        morphed_image = ImageOperations.morph_open_erode_close_image(copied_image)
        x1, x2, y1, y2 = self.get_ox_coordinates(morphed_image)
        height, width = ImageOperations.get_shape(morphed_image)

        if maskPosition is MaskPosition.BOTTOM:
            return copied_image[y1:height, x1:width]
        if maskPosition is MaskPosition.TOP:
            return copied_image[0:y2, 0:x2]

    def get_ox_coordinates(self, image):
        image = image.copy()
        morphed_image = ImageOperations.morph_open_erode_close_image(image)
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

    def remove_ox(self, image):
        copied_image = image.copy()
        gray = cv2.cvtColor(copied_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (0, 0, 0), 2)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
        result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        return image

    def remove_vertical_lines(self, image):
        copied_image = image.copy()
        copied_image = cv2.cvtColor(copied_image, cv2.COLOR_BGR2GRAY)
        copied_image = cv2.bitwise_not(copied_image)
        bw = cv2.adaptiveThreshold(copied_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        rows = bw.shape[0]
        verticalSize = int(rows / 30)

        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))

        vertical = cv2.erode(bw, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)

        smooth = np.copy(vertical)
        smooth = cv2.blur(smooth, (2, 2))
        (rows, cols) = np.where(edges != 0)
        vertical[rows, cols] = smooth[rows, cols]

        return vertical

    def delete_text(self, image):
        image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Every color except white
        lower = np.array([0, 42, 0])
        higher = np.array([179, 255, 255])

        # preparing the mask to overlay
        mask = cv2.inRange(hsv, lower, higher)

        image[mask > 0] = (255, 255, 255)

        return cv2.bitwise_not(image, image, mask=mask)