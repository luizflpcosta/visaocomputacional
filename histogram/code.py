import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

# Lê a imagem em tons de cinza
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

# determina o cinza mais frequente
cinza_mais_frequente = np.argmax(hist)
print(f'O nível de cinza mais frequente é: {cinza_mais_frequente}')

# define um valor de threshold adequado (de acordo com o histograma)
threshold_value = 120
_, img_bin = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

#exibe a imagem original e binária lado a lado
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title('Imagem Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Imagem binária (T={threshold_value})')
plt.imshow(img_bin, cmap='gray')
plt.axis('off')
plt.show()
