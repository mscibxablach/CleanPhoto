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
        image = image.copy()
        x1, x2, y1, y2 = self.get_ox_coordinates(image)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 5)

        return image

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