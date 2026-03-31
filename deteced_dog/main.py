import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("dogs.mp4")

# class động vật trong COCO
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]  
# person=0 → animal bắt đầu từ 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=ANIMAL_CLASSES)

    frame = results[0].plot()

    cv2.imshow("Animal Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()