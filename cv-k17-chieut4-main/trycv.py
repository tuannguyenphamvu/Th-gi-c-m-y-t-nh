import cv2

cap = cv2.VideoCapture(0) # mở camera 
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while(True):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key ==ord('s'):
        cv2.imwrite('saved_frame.png', frame)
        break

cap.release()
cv2.destroyAllWindows()