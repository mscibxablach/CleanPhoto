import cv2
import numpy as np
from random import randrange
from matplotlib import pyplot as plt
from MaskPosition import MaskPosition
from ImageOperations import ImageOperations


class PlotBoundDetector:

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def get_plot_bound(self, image):
        image = image.copy()

        image = self.image_processor.delete_text(image)
        cv2.imwrite('result_doc.jpg', image)
        bottom, top = self.image_processor.split_image(image)

        cv2.imwrite('bottom_result.jpg', bottom)
        cv2.imwrite('top_result.jpg', top)
        bottom = self.__prepare_image(bottom)
        top = self.__prepare_image(top)

        # cv2.imshow('top', top)
        # cv2.imshow('bottom', bottom)

        top_chunks = self.__get_chunks(top)
        bottom_chunks = self.__get_chunks(bottom)

        top_chunks, bottom_chunks = self.remove_reflexes(top_chunks, bottom_chunks)
        top_chunks = self.merge_nearest(top_chunks, 10)
        bottom_chunks = self.merge_nearest(bottom_chunks, 10)
        top_chunks = self.filter_chunks(top_chunks)
        bottom_chunks = self.filter_chunks(bottom_chunks)

        # self.draw_chunks(top_chunks, top)
        # self.draw_chunks(bottom_chunks, bottom)

        return top_chunks, bottom_chunks

    def remove_reflexes(self, top_chunks, bottom_chunks):
        bottom_chunks_to_remove = []
        top_chunks_to_remove = []

        for top_chunk in top_chunks:
            for bottom_chunk in bottom_chunks:
                if self.is_reflex(top_chunk, bottom_chunk):
                    if bottom_chunk not in bottom_chunks_to_remove:
                        bottom_chunks_to_remove.append(bottom_chunk)

        for chunks_to_remove in bottom_chunks_to_remove:
            bottom_chunks.remove(chunks_to_remove)

        for bottom_chunk in bottom_chunks:
            for top_chunk in top_chunks:
                if self.is_reflex(bottom_chunk, top_chunk):
                    if top_chunk not in top_chunks_to_remove:
                        top_chunks_to_remove.append(top_chunk)

        for chunks_to_remove in top_chunks_to_remove:
            top_chunks.remove(chunks_to_remove)

        return top_chunks, bottom_chunks

    def is_reflex(self, top_chunk, bottom_chunk):
        k = 10
        return top_chunk[0] - k < bottom_chunk[0] and top_chunk[-1] + k > bottom_chunk[-1]

    def __prepare_image(self, image):
        image = image.copy()
        image = self.image_processor.remove_ox(image)
        gray = ImageOperations.convert_to_gray(image)

        manual_thresholded = ImageOperations.color_threshold(gray)
        bilateral_filtered = cv2.bilateralFilter(manual_thresholded.copy(), 11, 75, 75)
        _, thresholded = cv2.threshold(bilateral_filtered, 70, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresholded, kernel, iterations=1)

        return dilated

    def __get_chunks(self, image):
        histogram = self.get_histogram(image, '')

        continous_chunks = self.get_continous_chunks(histogram)

        return continous_chunks

    def get_histogram(self, image):
        result = []
        rows, cols = ImageOperations.get_shape(image)

        for j in range(cols):
            sum = 0
            for i in range(rows):
                sum = sum + image[i, j]
            result.append(sum)

        return result

    def get_continous_chunks(self, array):
        result = []
        min_value = min(array)
        i = 0
        while i < len(array) - 1:
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
                    result.append(chunk)
                    i = j
                    continue
            elif value <= min_value:
                i += 1
                continue
        return result

    def filter_chunks(self, chunks, min_length_chunk=6):
        if len(chunks) > 0:
            if len(chunks) > 1:
                result = list(filter(lambda x: len(x) > min_length_chunk, chunks))
                return result
            else:
                return chunks
        else:
            return chunks

    def get_avrage_length(self, chunks):
        sum_len = sum(len(c) for c in chunks)
        chunks_count = len(chunks)
        return round(sum_len / chunks_count)

    def merge_nearest(self, chunks, distance):
        something_merged = False

        counter = 0
        while counter < len(chunks):
            has_next_chunk = counter + 1 < len(chunks)

            if not has_next_chunk:
                counter += 1
                continue

            current_chunk = chunks[counter]
            next_chunk = chunks[counter + 1]

            if self.can_merge(current_chunk, next_chunk, distance):
                chunks[counter] = list(range(current_chunk[0], next_chunk[-1] + 1))
                del chunks[counter + 1]
                something_merged = True
            counter += 1

        if something_merged:
            return self.merge_nearest(chunks, distance)

        return chunks

    def can_merge(self, first_chunk, second_chunk, k):
        end_first = first_chunk[-1]
        start_second = second_chunk[0]

        return (start_second - end_first) <= k

    @staticmethod
    def draw_chunks(chunks, image, position):
        width, height = ImageOperations.get_shape(image)

        for chunk in chunks:
            color = list(map(lambda x: int(x), list(np.random.choice(range(256), size=3))))
            start_column = chunk[0]
            end_column = chunk[-1]

            if position == MaskPosition.TOP:
                cv2.line(image, (start_column, 0), (start_column, 100), color, 1)
                cv2.line(image, (end_column, 0), (end_column, 100), color, 1)
            if position == MaskPosition.BOTTOM:
                cv2.line(image, (start_column, 100), (start_column, height), color, 1)
                cv2.line(image, (end_column, 100), (end_column, height), color, 1)




