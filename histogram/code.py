import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
img = cv2.imread('penguins.png', cv2.IMREAD_GRAYSCALE)
frame = img.copy()

# calculando e mostrando o histograma
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure()
plt.title("Histograma")
plt.xlabel("Intensidade do Pixel")
plt.ylabel("Número de Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

cinza_mais_frequente = np.argmax(hist)
print(f'O nível de cinza mais frequente é: {cinza_mais_frequente}')

limiar = 120
_, img_bin = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY)
cv2.imshow('Imagem binária', img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_logica = img_bin // 255
print('imagem binária lógica (0 e 1):')



