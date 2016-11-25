import cv2
# import csv
import numpy as np

# 359*255/360
lower_red1 = np.array([0, 45, 30], dtype="uint8")
upper_red1 = np.array([15, 245, 235], dtype="uint8")
lower_red2 = np.array([170, 45, 30], dtype="uint8")
upper_red2 = np.array([255, 245, 235], dtype="uint8")
image_path = "./images_train/25/00009.ppm"
image = cv2.imread(image_path)
img = cv2.GaussianBlur(np.copy(image), (5, 5), 0)
# img = cv2.blur(np.copy(image), (5, 5))
# img = cv2.medianBlur(np.copy(image), 5)
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    circles = cv2.HoughCircles(hsv_mask, cv2.HOUGH_GRADIENT, 1, 20,
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

    mser = cv2.MSER_create()
    regions = mser.detect(hsv_mask, None)
    for p in regions:
        print(p.pt)

    #hulls = [cv2.convexHull(p.pt) for p in regions]

    if cv2.waitKey(0) == 27:
        break
