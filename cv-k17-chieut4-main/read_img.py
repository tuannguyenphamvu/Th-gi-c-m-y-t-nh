import cv2 
img = cv2.imread('saved_frame.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # chuyển ảnh màu sang ảnh xám
gray_img[150:500,400:900] = 0 
cv2.imshow('image', gray_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
