# -----------------------------------------------------------------------------
# ------ NEURAL NETWORK SIGNS POTATOER ----------------------------------------
# ------ Basic OpenCV Neural Network that recognizes transit signals ----------
# ------ Por: Camilo A. Sampedro camilo.sampedro@udea.edu.co ------------------
# ------      Estudiante ingeniería de sistemas, Universidad de Antioquia -----
# ------      CC 1037640884 ---------------------------------------------------
# ------ Por: C. Vanessa Pérez cvanessa.perez@udea.edu.co ---------------------
# ------      Estudiante ingeniería de sistemas, Universidad de Antioquia -----
# ------      CC 1128440531 ---------------------------------------------------
# ------ Curso Básico de Procesamiento de Imágenes y Visión Artificial --------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# - Needed libraries ----------------------------------------------------------
# -----------------------------------------------------------------------------
import numpy as np                    # Array manipulation
import cv2                            # Image processing
from Color import Color               # Color store (Global values)
from sklearn.externals import joblib  # Number recognizer SVM reader
from skimage.feature import hog       # Hog features extraction


# -----------------------------------------------------------------------------
# - ImageRecognizer class -----------------------------------------------------
# -   Performs characteristic extraction operations with a given image     ----
# -----------------------------------------------------------------------------
class ImageRecognizer(object):
    # Canny edge circle detection size
    kernel_size = 3
    # Low threshold for detecting circles
    lowThreshold = 50
    # High threshold for detecting circles
    maxThreshold = 100
    # Ratio of the edge recognizion with Canny
    ratio = 3

    # Number file reader (Inside numbers/ folder)
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

    # Load the previously created number classifier
    clf = joblib.load("number_classifier.pkl")

    # Get the number image based on the parameter number
    #   number: Number to be searched on the numbers collection
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
    #   image_path: String with the path to the image in the file system
    @staticmethod
    def read_image(image_path):
        # Use CV2 canonical image reading
        image_read = cv2.imread(image_path)
        return image_read

    # Binarize a gray image
    #   image_gray: Grayish image (With just a channel, which is gray)
    @staticmethod
    def binary_of(image_gray):
        # Process a dynamic binarization
        (thresh, im_bw) = cv2.threshold(image_gray, 128, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return im_bw

    # Constructor
    #   image_path: Path to the image, for loading and processing
    def __init__(self, image_path):
        print("Analyzing %s" % image_path)
        # Save the image path
        self.image_path = image_path
        # Perform a first image reading
        self.raw_image = ImageRecognizer.read_image(self.image_path)
        # Convert to gray
        self.gray_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        # Equalize the image (For "normalizing" its colors)
        self.equalized_gray = cv2.equalizeHist(self.gray_image,
                                               self.gray_image)

    # Characteristic extraction, calls other methods for extracting those
    #  characteristics
    #   returns: Characteristic array: [isRed, hasRedCircle, numberInside]
    def extract_characteristics(self):
        # Is it a significant amount of red
        is_red = self.has_color(Color.RED)
        # Exist a circle inside the image
        has_red_circle = self.has_circle(Color.RED)
        # Use the SVM to extract and recognize the numbers
        number_inside = self.extract_numbers()
        # One hot codification for the numbers
        numbers_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Turn on the recognized numbers into the one hot
        for number in number_inside:
            numbers_one_hot[number] = 1
        # Return a full list of the characteristics
        return [is_red, has_red_circle] + numbers_one_hot

    # Extract a color from the image
    def extract_color(self, color):
        # Perform a image blurring for reducing color noise
        img = cv2.GaussianBlur(np.copy(self.raw_image), (5, 5), 0)
        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Get the image HSV boundaries (See Color class)
        bounds = Color.get(color)
        # Extract a mask for every pair of boundaries in bounds
        masks = [cv2.inRange(hsv_image, bound[0], bound[1])
                 for bound in bounds]
        # If there are two masks
        if len(masks) == 2:
            # Merge both mask into a single mask
            mask = np.empty_like(masks[0])
            mask = cv2.bitwise_or(masks[0], masks[1], mask)
            return mask
        # If there is only one
        elif len(masks) == 1:
            # Return it
            return masks[0]
        else:
            # Program is not prepared for anything else than 2 or 1 masks
            print("More than 3 masks!, I am not prepared to do that")
            return None

    # Detect if there is a significant amount of the color on the image
    #   color: String with the color, it MUST match a value in Color class
    #   return True if there is a big amount of that color or False else
    def has_color(self, color):
        # Get the image size for comparing the found color amount
        image_height, image_width = self.raw_image.shape[:2]
        # Calculate the area
        image_area = image_width * image_height
        # Extract the color region
        color_region = self.extract_color(color)
        # Count the area of the color region
        region_count = np.count_nonzero(color_region)
        # Compare with the image area. If it is at least a 10% of the original
        # image then it is big enough for True to be returned
        return region_count > image_area * 0.1

    # Extract circle from the image
    #   mask: Mask with the information for detecting circles (Binary image)
    def extract_circle(self, mask):
        # Detect edges using Canny for better circle recognizing
        detected_edges = cv2.Canny(mask, ImageRecognizer.lowThreshold,
                                   ImageRecognizer.lowThreshold
                                   * ImageRecognizer.ratio,
                                   ImageRecognizer.kernel_size)
        # Apply HoughCircles to the edges, it will create a circles array with
        # the radius and centers of them
        circles = cv2.HoughCircles(detected_edges, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0,
                                   maxRadius=0)
        return circles

    # Detect if there is a circle on the image
    #   color: Color of the circle. It MUST be a color in Color class
    #   return: True if there is at least one circle on the image, false if
    #    there's not
    def has_circle(self, color):
        # Extract a mask for that color
        mask = self.extract_color(color)
        # Extract the circles for that image
        circles = self.extract_circle(mask)
        # Check if the circles is empty (Or nothing is returned in this case)
        has_circle = circles is not None
        return has_circle

    # Extract the numbers using precreated SVM
    #   return: List of recognized numbers
    def extract_numbers(self):
        print("Extracting numbers")
        # Extract black color
        mask = self.extract_color(Color.BLACK)
        # Find the contours of that mask
        ctrs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
        # Create a bounding box for every single contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        # Recognized numbers array
        numbers = []
        # Iterate over the bounding box for applying the detection
        for rect in rects:
            # If it is big enough (Non-empty at least)
            if(abs(rect[2] - rect[0]) >= 1 and abs(rect[3] - rect[1]) >= 1):
                # Extract the points
                pt1_x = rect[0]
                pt1_y = rect[1]
                pt2_x = rect[2]
                pt2_y = rect[3]

                # Get the right cutting order
                if pt2_x > pt1_x and pt2_y > pt1_y:
                    roi = mask[pt1_x - 1:pt2_x + 1, pt1_y - 1:pt2_y + 1]
                elif pt2_x > pt1_x and pt2_y < pt1_y:
                    roi = mask[pt1_x - 1:pt2_x + 1, pt2_y - 1:pt1_y + 1]
                elif pt2_x < pt1_x and pt2_y > pt1_y:
                    roi = mask[pt2_x - 1:pt1_x + 1, pt1_y - 1:pt2_y + 1]
                else:
                    roi = mask[pt2_x - 1:pt1_x + 1, pt2_y - 1:pt1_y + 1]
                # Resize the box to the right SVM input size (28x28)
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                # Dilate the gotten image
                roi = cv2.dilate(roi, (3, 3))
                # Extract the hog characteristics
                roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14),
                                 cells_per_block=(1, 1), visualise=False)
                # Perform the prediction of the number
                nbr = ImageRecognizer.clf.predict(
                    np.array([roi_hog_fd], 'float64'))
                # Append the found number to the numbers array
                numbers.append(nbr)
        return numbers
