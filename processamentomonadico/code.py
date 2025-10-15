import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ajusta o diretório
os.chdir(os.path.dirname(__file__))

# lê imagem
img_int = cv2.imread('lena.pgm', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', img_int)
cv2.waitKey(0)

# converte para double (0 a 1)
img_double = img_int.astype(np.float64) / 255.0
print(img_double.dtype)

# brigthness para uint8 e double
valor = 63.75
brightness_int = cv2.add(img_int, np.array([valor]))
cv2.imshow('Brightness uint8', brightness_int)
cv2.waitKey(0)

valor = 0.25
brightness_double = np.clip(img_double + valor, 0, 1)
cv2.imshow('Brightness double', brightness_double)
cv2.waitKey(0)
# contrast para uint8 e double
valor1 = 2
contrast_int = cv2.multiply(img_int, np.array([valor1]))
# se o pixel ficar menor que 0 é 0, se ultrapassar 255 é 255
contrast_int = np.clip(contrast_int, 0, 255).astype(np.uint8)
cv2.imshow('Contrast uint8', contrast_int)
cv2.waitKey(0)

valor1 = 2
contrast_double = np.clip(img_double * valor1, 0, 1)
cv2.imshow('Contrast double', contrast_double)
cv2.waitKey(0)


# negative para uint8 e double
negative_int = 255 - img_int
cv2.imshow('Negative uint8', negative_int)
cv2.waitKey(0)

negative_double = 1.0 - img_double
cv2.imshow('Negative double', negative_double)
cv2.waitKey(0)

# posterização
niveis = 8

N = 256 // niveis
posterizacao_int = (img_int //N) * N
cv2.imshow(f'Posterizacao uint8 ({niveis} niveis)', posterizacao_int)
cv2.waitKey(0)

N = 1.0 / niveis
# normalização
posterizacao_double = np.floor(img_double / N) * N
cv2.imshow(f'Posterizacao double ({niveis} niveis)', posterizacao_double)
cv2.waitKey(0)

cv2.destroyAllWindows()

plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.title('Imagem Original')
plt.imshow(img_int, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title('Brilho aplicado')
plt.imshow(brightness_int, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title('Contraste aplicado')
plt.imshow(contrast_int, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title('Negativo aplicado')
plt.imshow(negative_int, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title('Posterisation aplicado')
plt.imshow(posterizacao_int, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()