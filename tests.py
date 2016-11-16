import cv2
import numpy as np

# 359*255/360
lower_red1 = np.array([0, 50, 20])
upper_red1 = np.array([20, 216, 216])
lower_red2 = np.array([240, 50, 20])
upper_red2 = np.array([255, 216, 216])
image_path = "./images_train/25/00014.ppm"
image = cv2.imread(image_path)

while True:
    cv2.imshow(image_path, image)

    img = cv2.GaussianBlur(np.copy(image), (5, 5), 0)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow(image_path + "HSV", hsv_image)
    print(hsv_image[50])

    hsv_mask_1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    hsv_mask_2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    cv2.imshow(image_path + "1", hsv_mask_1)
    cv2.imshow(image_path + "2", hsv_mask_2)
    if cv2.waitKey(0) == 27:
        break
