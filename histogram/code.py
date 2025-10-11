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