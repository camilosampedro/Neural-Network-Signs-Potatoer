import numpy as np
import cv2
import Color


class ImageRecognizer():

    kernel_size = 3
    lowThreshold = 50
    maxThreshold = 100
    ratio = 3

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
        is_red = self.has_color(Color.RED)
        return [is_red]

    def extract_shapes(self):
        print("Pending extract_shapes")

    def extract_color(self, color):
        img = cv2.GaussianBlur(np.copy(self.raw_image), (5, 5), 0)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bounds = Color.get(color)
        masks = [cv2.inRange(hsv_image, bound[0], bound[1])
                 for bound in bounds]
        if len(masks) == 2:
            mask = np.empty_like(masks[0])
            mask = cv2.bitwise_or(masks[0], masks[1], mask)
        elif len(masks) == 1:
            mask = masks[1]
        else:
            print("More than 3 masks!, I am not prepared to do that")
            return None

    def has_color(self, color):
        color_region = self.extract_color(color)
        print("Pending has_color")

    def extract_circle(self, mask):
        detected_edges = cv2.Canny(mask, ImageRecognizer.lowThreshold,
                                   ImageRecognizer.lowThreshold
                                   * ImageRecognizer.ratio,
                                   ImageRecognizer.kernel_size)
        circles = cv2.HoughCircles(detected_edges, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0,
                                   maxRadius=0)
        return circles
