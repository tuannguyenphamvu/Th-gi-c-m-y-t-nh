import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
face_reg = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc video")
        break

    if frame is None:
        bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dw = cv.GaussianBlur(bw, (5, 5), 0)
        
