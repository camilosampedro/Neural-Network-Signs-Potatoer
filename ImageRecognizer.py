import numpy as np
import cv2
from Color import Color
from sklearn.externals import joblib
from skimage.feature import hog


class ImageRecognizer(object):
    kernel_size = 3
    lowThreshold = 50
    maxThreshold = 100
    ratio = 3
    number_0 = cv2.imread("numbers/0.bmp", 0)
    number_1 = cv2.imread("numbers/1.bmp", 0)
    number_2 = cv2.imread("numbers/2.bmp", 0)
    number_3 = cv2.imread("numbers/3.bmp", 0)
    number_4 = cv2.imread("numbers/4.bmp", 0)
    number_5 = cv2.imread("numbers/5.bmp", 0)
    number_6 = cv2.imread("numbers/6.bmp", 0)
    number_7 = cv2.imread("numbers/7.bmp", 0)
    number_8 = cv2.imread("numbers/8.bmp", 0)
    number_9 = cv2.imread("numbers/9.bmp", 0)
    clf = joblib.load("number_classifier.pkl")

    @staticmethod
    def get_number(number):
        return ImageRecognizer.binary_of({
            0: ImageRecognizer.number_0,
            1: ImageRecognizer.number_1,
            2: ImageRecognizer.number_2,
            3: ImageRecognizer.number_3,
            4: ImageRecognizer.number_4,
            5: ImageRecognizer.number_5,
            6: ImageRecognizer.number_6,
            7: ImageRecognizer.number_7,
            8: ImageRecognizer.number_8,
            9: ImageRecognizer.number_9,
        }.get(number, None))

    # Read a image
    #  Params:
    #   * image_path - String with the path to the image in the file system
    @staticmethod
    def read_image(image_path):
        # Use CV2 canonical image reading
        image_read = cv2.imread(image_path)
        return image_read

    @staticmethod
    def binary_of(image_gray):
        (thresh, im_bw) = cv2.threshold(image_gray, 128, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return im_bw

    def __init__(self, image_path):
        print("Analyzing %s" % image_path)
        self.image_path = image_path
        self.raw_image = ImageRecognizer.read_image(self.image_path)

    # Characteristic array: [isRed, hasRedCircle, numberInside]
    def extract_characteristics(self):
        is_red = self.has_color(Color.RED)
        has_red_circle = self.has_circle(Color.RED)
        number_inside = self.extract_numbers()
        return [is_red, has_red_circle, number_inside]

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
            return mask
        elif len(masks) == 1:
            return masks[0]
        else:
            print("More than 3 masks!, I am not prepared to do that")
            return None

    def has_color(self, color):
        image_height, image_width = self.raw_image.shape[:2]
        image_area = image_width * image_height
        color_region = self.extract_color(color)
        region_count = np.count_nonzero(color_region)
        return region_count > image_area * 0.1

    def extract_circle(self, mask):
        detected_edges = cv2.Canny(mask, ImageRecognizer.lowThreshold,
                                   ImageRecognizer.lowThreshold
                                   * ImageRecognizer.ratio,
                                   ImageRecognizer.kernel_size)
        circles = cv2.HoughCircles(detected_edges, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0,
                                   maxRadius=0)
        return circles

    def has_circle(self, color):
        mask = self.extract_color(color)
        circles = self.extract_circle(mask)
        return circles is not None

    def extract_numbers(self):
        print("Extracting numbers")
        mask = self.extract_color(Color.BLACK)
        ctrs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        for rect in rects:
            print(rect)
            cv2.rectangle(self.raw_image, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          (0, 255, 0), 3)
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = mask[pt1:pt1 + leng, pt2:pt2 + leng]
            cv2.imshow("roi", roi)
            cv2.waitKey()
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14),
                             cells_per_block=(1, 1), visualise=False)
            nbr = ImageRecognizer.clf.predict(
                np.array([roi_hog_fd], 'float64'))
            cv2.putText(self.raw_image, str(int(nbr[0])), (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("Resulting Image with Rectangular ROIs", self.raw_image)
        cv2.waitKey()
        print("Pending extract_numbers")
