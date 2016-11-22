import cv2
import csv
import numpy as np

# 359*255/360
lower_red1 = np.array([0, 40, 15], dtype="uint8")
upper_red1 = np.array([15, 245, 245], dtype="uint8")
lower_red2 = np.array([170, 40, 15], dtype="uint8")
upper_red2 = np.array([255, 245, 245], dtype="uint8")
image_path = "./images_train/07/00033.ppm"
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

    cv2.imshow("m1", hsv_mask_1)
    cv2.imshow("m2", hsv_mask_2)
    if cv2.waitKey(0) == 27:
        break
