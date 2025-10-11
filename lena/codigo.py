import cv2
import numpy as np
import time
import os

os.chdir(os.path.dirname(__file__))

img = cv2.imread('lena.pgm', cv2.IMREAD_GRAYSCALE)
frame = img.copy()

x1, y1, x2, y2 = 240, 230, 290, 260
regiao_selecionada = frame[y1:y2, x1:x2].copy()

for i in range(3):
    for d in range(0, 15, 2):
        temp = frame.copy()
        temp[y1+d:y2+d, x1:x2] = regiao_selecionada
        cv2.imshow('Lena Piscando', temp)
        cv2.waitKey(30)
    for d in range(15, 0, -2):
        temp = frame.copy()
        temp[y1+d:y2+d, x1:x2] = regiao_selecionada
        cv2.imshow('Lena Piscando', temp)
        cv2.waitKey(30)
    time.sleep(0.5)

cv2.destroyAllWindows()