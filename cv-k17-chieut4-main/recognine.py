import cv2 as cv
import numpy as np
import os
recog_tool = cv.face.LBPHFaceRecognizer_create()
recog_tool.read("face_recognizer_model.xml")
labels_dict = np.load("labels_dict.npy", allow_pickle=True).item()
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
while True:
 ret, frame = cap.read()
 if not ret:
  print("Failed to grab frame")
  break
 if frame is not None:
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  dgray = cv.GaussianBlur(gray, (5, 5), 0)
  faces = face_cascade.detectMultiScale(dgray, 1.3, 5)

  for (x, y, w, h) in faces:
   face_img = dgray[y:y+h, x:x+w]
   name, dotincay = recog_tool.predict(face_img)
   if dotincay < 80:
    name = labels_dict[name]
    color = (0,255,0)
   else:
     name = "Unknown"
     color = (0,0,255)
     cv.rectangle(frame, (x,y), (x+w, y+h), color, 2)
   cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
   cv.imshow("Face Recognition", frame)
   if cv.waitKey(1) & 0xFF == ord("q"):
    break
   cap.release()
cv.destroyAllWindows()