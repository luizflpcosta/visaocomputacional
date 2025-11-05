import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))

img = cv2.imread('flowers9.png')
cv2.imshow('Imagem Original', img)
cv2.waitKey(0)

# converte de bgr (padrão opencv) para rgb
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

cores = ('r', 'g', 'b')
plt.figure(figsize=(10, 5))
for i, cor in enumerate(cores):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=cor)
    plt.xlim([0, 256])
plt.title('Histograma de Cores')
plt.xlabel('Intensidade de Cor')
plt.ylabel('Número de Pixels')
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Imagem RGB')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(R, cmap='Reds')
plt.title('Canal Vermelho (R)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(G, cmap='Greens')
plt.title('Canal Verde (G)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(B, cmap='Blues')
plt.title('Canal Azul (B)')
plt.axis('off')

plt.tight_layout()
plt.show()