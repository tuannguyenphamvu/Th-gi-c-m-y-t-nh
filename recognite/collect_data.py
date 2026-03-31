import cv2 as cv
import os
import time
import numpy as np

# ========== CONFIG ==========
DATA_DIR = "data"
NUM_SAMPLES = 100
IMG_SIZE = (200, 200)
DELAY_MS = 150

# ========== DNN MODEL PATH ==========
PROTO = "models/deploy.prototxt"
MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"

 # DNN params
CONF_THRES = 0.6   # tăng lên 0.7 nếu bắt nhầm

PADDING = 0.20     # nới bbox để lấy đủ trán/cằm

SHOW_CROP_PREVIEW = False


def detect_faces_dnn(net, frame, conf_thres=0.6):
    """Return list of (x, y, w, h, conf) in pixel coords."""
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()

    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        boxes.append((x1, y1, bw, bh, conf))
    return boxes


def pad_box(x, y, w, h, img_w, img_h, pad=0.2):
    """Expand bbox by pad% and clamp to image."""
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w - 1, x + w + px)
    y2 = min(img_h - 1, y + h + py)
    return x1, y1, x2 - x1, y2 - y1


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    user_name = input("Enter your name: ").strip()
    if not user_name:
        print("Empty name!")
        return

    save_path = os.path.join(DATA_DIR, user_name)
    if os.path.exists(save_path):
        print("User already exists. Please choose a different name.")
        return
    os.makedirs(save_path, exist_ok=True)

    # load DNN
    if not os.path.exists(PROTO) or not os.path.exists(MODEL):
        print("Missing DNN files! Make sure you have:")
        print(" -", PROTO)
        print(" -", MODEL)
        return

    net = cv.dnn.readNetFromCaffe(PROTO, MODEL)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    dem = 0
    last_save_ms = 0

    print(f"Collecting {NUM_SAMPLES} images for: {user_name}")
    print("Tips: nhìn thẳng, xoay nhẹ trái/phải, lên/xuống. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w = frame.shape[:2]

        # detect with DNN
        boxes = detect_faces_dnn(net, frame, CONF_THRES)

        cv.putText(frame, f"faces={len(boxes)}", (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if len(boxes) > 0:
            # lấy bbox lớn nhất (theo area)
            x, y, bw, bh, conf = max(boxes, key=lambda b: b[2] * b[3])

            # nới bbox cho đủ trán/cằm
            x, y, bw, bh = pad_box(x, y, bw, bh, w, h, PADDING)

            # vẽ bbox
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.putText(frame, f"{conf:.2f}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now_ms = time.time() * 1000
            if dem < NUM_SAMPLES and (now_ms - last_save_ms) >= DELAY_MS:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                face_img = gray[y:y + bh, x:x + bw]
                face_img = cv.resize(face_img, IMG_SIZE)

                dem += 1
                out_path = os.path.join(save_path, f"{dem:04d}.jpg")
                cv.imwrite(out_path, face_img)

                last_save_ms = now_ms

                if SHOW_CROP_PREVIEW:
                    cv.imshow("Face Crop Preview", face_img)

        cv.putText(frame, f"Saved: {dem}/{NUM_SAMPLES}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Collect Data (DNN)", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or dem >= NUM_SAMPLES:
            break

    cap.release()
    cv.destroyAllWindows()
    print("Done. Saved to:", save_path)


if __name__ == "__main__":
    main()