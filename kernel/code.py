import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))

# carrega imagem em escala cinza
img = cv2.imread('flowers4.png', cv2.IMREAD_GRAYSCALE)
img_double = img.astype(np.float64) / 255.0

cv2.imshow('Imagem Original', img)
cv2.waitKey(0)

kU = np.ones((15, 15), dtype=np.float64) 
kU = kU / kU.sum()

plt.figure(figsize=(4, 4))
plt.imshow(kU, cmap='gray')
plt.title('Kernel Médio 15x15 (volume unitário)')
plt.colorbar()
plt.show()


imU = cv2.filter2D(img_double, -1, kU)

cv2.imshow('Imagem após Convolução (Kernel Médio 15x15)', imU)
cv2.waitKey(0)

sigma = 5
L = 8
ksize = (2* L) + 1
g1d = cv2.getGaussianKernel(ksize, sigma)
kG = g1d @ g1d.T

plt.figure(figsize=(10, 5))
plt.imshow(kG, cmap='gray')
plt.title('Kernel Gaussiano 17x17 (σ=5)')
plt.show()

imG = cv2.GaussianBlur(img_double, (ksize, ksize), sigma)
cv2.imshow('Imagem após Convolução (Kernel Gaussiano σ=5)', imG)
cv2.waitKey(0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_double, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imU, cmap='gray')
plt.title('Média 15x15')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imG, cmap='gray')
plt.title('Gaussiano σ=5')
plt.axis('off')

plt.show()


print("""
Observações:
- O filtro médio (imU) é não isotrópico e pode gerar linhas verticais/horizontais.
- O filtro Gaussiano (imG) é isotrópico e produz uma suavização mais natural.
""")