import cv2 as cv
import numpy as np
import os

faces = []
labels = []
labels_dict = {}
curent_label = 0

for user in os.listdir(data_path):
    user_path = os.path.join(data_path, user)
    if not os.path.isdir(user_path):
        continue
    labels_dict[curent_label] = user
    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)
        img_bw = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        faces.append(img_bw)
        labels.append(curent_label)
    curent_label += 1
    
req_tool.train(faces, np.array(labels))

req_tool.save("face_recognizer_model.xml")

print("Đã train xong model nhận diện khuôn mặt và lưu vào file face_recognizer_model.xml")
        
    