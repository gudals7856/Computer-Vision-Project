import cv2
import numpy as np

img1_origin = cv2.imread('frame1.jpg')
img1 = cv2.resize(img1_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

height, width, _ = img1_origin.shape #720 1280
B, G, R = cv2.split(img1_origin)
rgb_data = np.zeros((256, 256, 256), np.uint8)
location_data = np.zeros((720, 1280), np.uint8)

for x in (0, 1280):
    for y in (0, 720):
        r = R[y, x]
        g = G[y, x]
        b = B[y, x]

cv2.imshow('result', img1_origin)
cv2.waitKey()
cv2.destroyAllWindows()