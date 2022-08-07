import matplotlib.pyplot as plt
import cv2
from scanner1 import get_rectangle
from py_utils import four_point_transform

mask = cv2.imread("scanner3Mask.png")
img_height = 500
img_path = "./test_imgs/test3.jpg"

img = cv2.imread(img_path)
assert img is not None, "Can't open {img_path}"
img = cv2.resize(img, None, fx=0.5, fy=0.5)
autoRect = get_rectangle(mask, img_height)

warpedImg = four_point_transform(img, autoRect)
cv2.imshow("scanner3", warpedImg)
cv2.waitKey(0)