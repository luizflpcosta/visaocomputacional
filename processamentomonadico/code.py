import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))

img_int = cv2.imread('lena.pgm', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', img_int)
cv2.waitKey(0)

img_double = img_int.astype(np.float64)
print(img_double.dtype)

valor = 50
brightness = cv2.add(img_int, np.array([valor]))
cv2.imshow('Brightness aplicado', brightness)
cv2.waitKey(0)