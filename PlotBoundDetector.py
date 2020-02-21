import cv2
import numpy as np
from matplotlib import pyplot as plt

from ImageOperations import ImageOperations


class PlotBoundDetector:
    def get_plot_bound(self, image):
        image = image.copy()
        gray = ImageOperations.convert_to_gray(image)

        manual_thresholded = ImageOperations.color_threshold(gray)
        bilateral_filtered = cv2.bilateralFilter(manual_thresholded.copy(), 11, 75, 75)
        _, thresholded = cv2.threshold(bilateral_filtered, 70, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresholded, kernel, iterations=1)

        histogram = self.get_histogram(dilated, '')

        continous_chunks = self.get_continous_chunks(histogram)
        continous_chunks = self.filter_chunks(continous_chunks)
        continous_chunks = self.merge_nearest(continous_chunks, 15)
        self.draw_chunks(continous_chunks, image)

        return image

    def get_histogram(self, image, title):
        result = []
        rows, cols = ImageOperations.get_shape(image)

        for j in range(cols):
            sum = 0
            for i in range(rows):
                sum = sum + image[i, j]
            result.append(sum)

        plt.plot(result)
        plt.title(title)
        plt.show()

        return result

    def get_continous_chunks(self, array):
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

    def filter_chunks(self, chunks):
        avg_len = self.get_avrage_length(chunks)
        result = list(filter(lambda x: len(x) >= avg_len, chunks))
        return result

    def get_avrage_length(self, chunks):
        sum_len = sum(len(c) for c in chunks)
        chunks_count = len(chunks)
        return round(sum_len / chunks_count)

    def merge_nearest(self, chunks, distance):
        result = []
        chunk_count = len(chunks)
        merged = False
        counter = 0
        inner_counter = 0
        while counter < (chunk_count - 1):
            start = chunks[counter][0]
            end = chunks[counter][-1]
            inner_counter = counter + 1
            while inner_counter < chunk_count:
                if self.can_merge(chunks[counter], chunks[inner_counter], distance):
                    end = chunks[inner_counter][-1]
                    inner_counter += 1
                    merged = True
                else:
                    break
            result.append(list(range(start, end + 1)))
            counter += 1

        if counter == 1 and not merged:
            result.append(chunks[-1])

        if counter == 0:
            return chunks

        return result

    @staticmethod
    def can_merge(first_chunk, second_chunk, k):
        end_first = first_chunk[-1]
        start_second = second_chunk[0]

        return (start_second - end_first) <= k

    def draw_chunks(self, chunks, image):
        width, height = ImageOperations.get_shape(image)

        for chunk in chunks:
            color = list(map(lambda x: int(x), list(np.random.choice(range(256), size=3))))
            start_column = chunk[0]
            end_column = chunk[-1]
            cv2.line(image, (start_column, 0), (start_column, height), color, 1)
            cv2.line(image, (end_column, 0), (end_column, height), color, 1)


