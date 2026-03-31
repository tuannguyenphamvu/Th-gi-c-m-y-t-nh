import cv2 as cv
import numpy as np
import os

data_path = "data"

IMG_SIZE = (200, 200)  # PHẢI GIỐNG collect_data

reg_tool = cv.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_dict = {}
current_label = 0

users = [u for u in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, u))]
users.sort()  # để label ổn định

for user in users:
    user_path = os.path.join(data_path, user)
    label_dict[current_label] = user

    imgs = [f for f in os.listdir(user_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    imgs.sort()

    for img in imgs:
        img_path = os.path.join(user_path, img)
        img_bw = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img_bw is None:
            continue

        # ✅ đảm bảo cùng kích thước
        img_bw = cv.resize(img_bw, IMG_SIZE)

        faces.append(img_bw)
        labels.append(current_label)

    print(f"Loaded {user}: {len(imgs)} images")
    current_label += 1

if len(faces) == 0:
    print("No training images found!")
    exit()

reg_tool.train(faces, np.array(labels))
reg_tool.save("face_recognizer_model.yml")

np.save("label_dict.npy", label_dict)

print("Model trained and saved successfully.")
print("label_dict:", label_dict)