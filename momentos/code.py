import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image = cv2.imread('aviao_ed.png')
vis = image.copy()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

contornos, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contorno in contornos:
    area = cv2.contourArea(contorno)
    print(f'área do objeto: {area} pixels quadrados')

    momentos = cv2.moments(contorno)

    if momentos['m00'] != 0:
        cx = int(momentos['m10'] / momentos['m00'])
        cy = int(momentos['m01'] / momentos['m00'])
        print(f'Centróide: ({cx}, {cy})')

        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    else:
        print('Centróide indefinido (contorno com área zero)')

    mu20 = momentos['mu20']
    mu02 = momentos['mu02']
    mu11 = momentos['mu11']

    print('\nMomentos Centrais:')
    print(f'μ20 = {mu20}')
    print(f'μ02 = {mu02}')
    print(f'μ11 = {mu11}')

    I = np.array ([
        [mu20, mu11],
        [mu11, mu02]
    ])
    print('\nMatriz de Inércia')
    print(I)

    vals, vecs = np.linalg.eig(I) # vals = autovalores 
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    lambda1, lambda2 = vals[0], vals[1]

    v1 = vecs[:, 0]
    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = np.degrees(angle_rad)

    sigma1 = np.sqrt(lambda1 / area)
    sigma2 = np.sqrt(lambda2 / area)

    a = 2.0 * sigma1
    b = 2.0 * sigma2
    

    center = (int(round(cx)), int(round(cy)))
    axes = (int(round(a)), int(round(b)))
    color = (0, 255, 0)
    thickness = 2

    cv2.circle(vis, center, 3, (0, 0, 255), -1)
    cv2.ellipse(vis, center, axes, angle_deg, 0, 360, color, thickness)

    print(f'Objeto: área={area:.0f}, centroide=({cx:.2f},{cy:.2f}), angle={angle_deg:.2f}')
    print(f"  mu20={mu20:.2f}, mu02={mu02:.2f}, mu11={mu11:.2f}")
    print(f"  lambda1={lambda1:.2f}, lambda2={lambda2:.2f}")
    print(f"  semi-eixos (a,b) = ({a:.2f}, {b:.2f})\n")
cv2.imshow('Elipse equivalente', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()