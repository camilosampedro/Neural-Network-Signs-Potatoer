import cv2
# import csv
import numpy as np
from ImageRecognizer import ImageRecognizer

img_rec = ImageRecognizer("./images_train/00/00002.ppm")
characteristics = img_rec.extract_characteristics()


# 359*255/360
lower_red1 = np.array([0, 45, 15], dtype="uint8")
upper_red1 = np.array([15, 245, 230], dtype="uint8")
lower_red2 = np.array([170, 45, 15], dtype="uint8")
upper_red2 = np.array([255, 245, 230], dtype="uint8")
lower_black = np.array([0, 0, 0], dtype="uint8")
upper_black = np.array([255, 255, 100], dtype="uint8")
image_path = "./images_train/00/00002.ppm"
image = cv2.imread(image_path)
image = cv2.equalizeHist(image)
img = cv2.GaussianBlur(np.copy(image), (5, 5), 0)
# img = cv2.blur(np.copy(image), (5, 5))
# img = cv2.medianBlur(np.copy(image), 5)
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

kernel_size = 3
lowThreshold = 50
maxThreshold = 100
ratio = 3

# with open('hsv.csv', 'w') as csvfile:
# spamwriter = csv.writer(csvfile, delimiter=',',
#                         quotechar='"', quoting=csv.QUOTE_MINIMAL)
# for (i) in hsv_image:
#     js = [str(v) for v in i]
#     # print(type(js))
#     # print(js)
#     spamwriter.writerow(js)
#     # for (j) in js:
#     # print(type(j))
#     # print(j)
#     # print('----------')
#     # spamwriter.writerow(str(j))
#     # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#     # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


while True:
    cv2.imshow(image_path, image)
    cv2.imshow(image_path + "HSV", hsv_image)
    # print(hsv_image[50])
    hsv_1 = np.empty_like(hsv_image)
    hsv_2 = np.empty_like(hsv_image)
    np.copyto(hsv_1, hsv_image)
    np.copyto(hsv_2, hsv_image)
    hsv_mask_1 = cv2.inRange(hsv_1, lower_red1, upper_red1)
    hsv_mask_2 = cv2.inRange(hsv_2, lower_red2, upper_red2)
    hsv_mask = np.empty_like(hsv_mask_1)
    hsv_mask = cv2.bitwise_or(hsv_mask_1, hsv_mask_2, hsv_mask)
    detected_edges = cv2.Canny(hsv_mask, lowThreshold, lowThreshold * ratio,
                               kernel_size)
    circles = cv2.HoughCircles(detected_edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    # ensure at least some circles were found
    if circles is not None:
        output = image.copy()
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5),
                          (0, 128, 255), -1)
        # show the output image
        cv2.imshow("output", np.hstack([image, output]))
    cv2.imshow("m1", hsv_mask_1)
    cv2.imshow("m2", hsv_mask_2)
    cv2.imshow("join", hsv_mask)
    cv2.imshow("de", detected_edges)

    # mser = cv2.MSER_create()
    # regions = mser.detect(hsv_mask, None)
    # for p in regions:
    #     print(np.array(p.pt))
    # points = np.array([np.array(p.pt) for p in regions])
    # hulls = cv2.convexHull(points)
    # hulls = [cv2.convexHull(np.array(p)) for p in regions]

    # contours = cv2.findContours(detected_edges, cv2.RETR_TREE,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    #
    # aprox = np.zeros(len(detected_edges))
    # for i in range(len(contours)):
    #     aprox = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True
    # )
    #                              * 0.02, True)
    #     print(aprox)
    # cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=15)
    # cv2.imshow("lados", image)
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    cv2.imshow("Black", black_mask)

    if cv2.waitKey(0) == 27:
        break
