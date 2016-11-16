import numpy as np
import cv2


class ImageRecognizer():
    # Read a image
    #  Params:
    #   * image_path - String with the path to the image in the file system
    @staticmethod
    def read_image(image_path):
        # Use CV2 canonical image reading
        image_read = cv2.imread(image_path)
        return image_read

    def __init__(self, image_path):
        print("Analyzing %s" % image_path)
