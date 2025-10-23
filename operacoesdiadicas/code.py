import cv2
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

video = cv2.VideoCapture('traffic_sequence.mpg')

if not video.isOpened():
    print('Erro ao abrir o vídeo')
    exit()

ret, prev_frame = video.read()
if not ret:
    print('Erro ao ler o primeiro frame')
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(prev_gray, gray)
    _, diff_thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('Frame Original', gray)
    cv2.imshow('Diferença Absoluta', diff_frame)
    cv2.imshow('Threshold', diff_thresh)

    prev_gray = gray.copy()

    if cv2.waitKey(30) & 0xFF == ord('q'): # APERTE Q PARA SAIR
        break

video.release()
cv2.destroyAllWindows()
