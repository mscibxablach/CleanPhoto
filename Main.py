import os
import cv2
import numpy as np
import json
from ImageOperations import ImageOperations
from ImageProcessor import ImageProcessor
from MaskPosition import MaskPosition
from PlotBoundDetector import PlotBoundDetector
from PlotBoundCalculator import PlotBoundCalculator

print(cv2.__version__)


def save_to_file_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def get_filename_from_path(path):
    return os.path.basename(path)


def get_all_images(path):
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(os.path.join(path,name))
    return files


def main_v3():
    image_path = 'Photos/0.jpg'

    image_processor = ImageProcessor()
    plot_bound_detector = PlotBoundDetector(image_processor)

    image = ImageOperations.read_image(image_path)

    top_chunks, bottom_chunks = plot_bound_detector.get_plot_bound(image)
    PlotBoundDetector.draw_chunks(top_chunks, image, MaskPosition.TOP)
    PlotBoundDetector.draw_chunks(bottom_chunks, image, MaskPosition.BOTTOM)
    cv2.imwrite('test.jpg', image)
    cv2.waitKeyEx()


def main_v2():
    results = []
    output_path = 'Result'
    image_paths = get_all_images('Photos')
    image_processor = ImageProcessor()
    plot_bound_detector = PlotBoundDetector(image_processor)
    for image_path in image_paths:
        print(image_path)
        image = ImageOperations.read_image(image_path)
        copied_image = image.copy()

        top_chunks, bottom_chunks = plot_bound_detector.get_plot_bound(image)
        PlotBoundDetector.draw_chunks(top_chunks, image, MaskPosition.TOP)
        PlotBoundDetector.draw_chunks(bottom_chunks, image, MaskPosition.BOTTOM)

        # ratio = PlotBoundCalculator.calculate_distance_ratio(top_result, bottom_result)
        # results.append((get_filename_from_path(image_path), ratio))
        # print(bottom_result)

        cv2.imwrite(os.path.join(output_path, get_filename_from_path(image_path)), image)
    # save_to_file_json(results, "test.json")


def main():
    main_v2()


if __name__ == '__main__':
    main()










