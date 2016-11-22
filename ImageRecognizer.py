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
        self.image_path = image_path
        self.raw_image = ImageRecognizer.read_image(self.image_path)

    def extract_characteristics(self):
        print("Pending extract_characteristics")

    def extract_shapes(self):
        print("Pending extract_shapes")

    def extract_color(self, color):
        img = cv2.GaussianBlur(np.copy(self.raw_image), (5, 5), 0)

    def has_color(self, color):
