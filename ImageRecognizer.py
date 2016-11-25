import numpy as np
import cv2
import Color


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
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color == Color.RED:
            hsv_mask_1 = cv2.inRange(hsv_1, lower_red1, upper_red1)
            hsv_mask_2 = cv2.inRange(hsv_2, lower_red2, upper_red2)

    def has_color(self, color):
