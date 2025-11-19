import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(__file__))

def get_xy_cromaticidade(image_bgr):
    # converte  a imagem BGR para o espaço XYZ
    image_xyz = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2XYZ)
    X, Y, Z = cv2.split(image_xyz.astype('float32'))
    sum_xyz = X + Y + Z
    # evitar divisão por zero para pixel preto ou muito escuro
    sum_xyz[sum_xyz == 0] = 1e-6
    x = X / sum_xyz
    y = Y / sum_xyz
    return x, y

image = cv2.imread('yellowtargets.png')

if image is None:
    print('Erro ao carregar a imagem')
else:
    x_plane, y_plane = get_xy_cromaticidade(image)

    cv2.imshow('Plano de Cromaticidade x', x_plane)
    cv2.imshow('Plano de Cromaticidade y', y_plane)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------ k - means #
h, w = x_plane.shape

# converte para vetor
pixels_xy = np.column_stack((x_plane.flatten(), y_plane.flatten()))

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels_xy)

labels = kmeans.labels_

cluster_image = labels.reshape(h, w)

plt.imshow(cluster_image, cmap='tab10')
plt.title('Clusters no plano de cromaticidade (K-means)')
plt.show()

centers = kmeans.cluster_centers_
plt.figure(figsize=(7,7))
plt.scatter(pixels_xy[:,0], pixels_xy[:,1], s=1, alpha=0.3, label='pixels')

plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X', label='Centroides')
plt.title("Plano xy com centros dos clusters")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

cluster_amarelo = np.argmax(centers[:, 0])

amarelo_mask = (labels == cluster_amarelo).astype(np.uint8)
binary_image = amarelo_mask.reshape(h, w)

plt.imshow(binary_image, cmap='gray')
plt.title("Máscara binária dos alvos amarelos")
plt.show()

kernel = np.ones((7,7), np.uint8)
binary_clean = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)

plt.imshow(binary_clean, cmap='gray')
plt.title("Máscara limpa")
plt.show()

num_labels, labels_cc = cv2.connectedComponents(binary_clean)


centroides = []
min_area = 500

for lab in range(1, num_labels):  # pula o fundo (label 0)
    ys, xs = np.where(labels_cc == lab)
    if len(xs) < min_area:
        continue
    
    cx = xs.mean()
    cy = ys.mean()
    centroides.append((cx, cy))

# Exibir centróides
print("\nCentróides encontrados:")
for i, (cx, cy) in enumerate(centroides, start=1):
    print(f"Alvo {i}: ({cx:.2f}, {cy:.2f})")

# 5. Opcional: mostrar imagem original com centróides desenhados
cv2.destroyAllWindows()
image_centroids = image.copy()
for (cx, cy) in centroides:
    cv2.circle(image_centroids, (int(cx), int(cy)), 10, (0,0,255), -1)

cv2.imshow("Centroides identificados", image_centroids)
cv2.waitKey(0)
cv2.destroyAllWindows()