import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

img = cv2.imread('penguins.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original', img)
cv2.waitKey(0)

img_blur = cv2.GaussianBlur(img, (3, 3), 0)

Kv = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Ku = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

Iv = cv2.filter2D(img_blur, cv2.CV_64F, Kv)
Iu = cv2.filter2D(img_blur, cv2.CV_64F, Ku)

I = np.sqrt(Iu**2 + Iv**2)

I = cv2.normalize(I, None, 0, 255, cv2.NORM_MINMAX)
I = np.uint8(I)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Gradiente horizontal (Iu)')
plt.imshow(np.abs(Iu), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Gradiente Vertical (Iv)')
plt.imshow(np.abs(Iv), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Imagem das Bordas (I = √(Iu² + Iv²))")
plt.imshow(I, cmap='gray')
plt.axis('off')

plt.show()