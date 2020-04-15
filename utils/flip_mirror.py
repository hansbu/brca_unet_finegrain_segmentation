import cv2
import numpy as np

img = cv2.imread('cat.jpg')

mirror = np.flip(img, 0)
vert = np.flip(img, 1)

cv2.imshow('mirror', mirror)
cv2.waitKey(0)


cv2.imshow('vert', vert)
cv2.waitKey(0)