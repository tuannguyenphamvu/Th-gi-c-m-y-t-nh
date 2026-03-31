import os
import time
from collections import deque

import cv2 as cv
import numpy as np
import smtplib
from email.mime.text import MIMEText


# ================== CONFIG ==================
IMG_SIZE = (200, 200)

# LBPH: conf càng nhỏ càng giống
# "Non hơn" => tăng ngưỡng lên (vd 70~85) nhưng nên dùng voting để an toàn
THRESHOLD = 75.0

OWNER_NAME = "A"
UNLOCK_COOLDOWN_SEC = 20

# Voting: mở khoá khi đúng >= REQUIRED_HITS trong WINDOW_FRAMES frame gần nhất
WINDOW_FRAMES = 7
REQUIRED_HITS = 4

# Face detection params
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (90, 90)

# Email config (khuyến nghị set ENV thay vì ghi thẳng)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "fullday2k5@gmail.com")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "iejx czjz tlki wcwy")   # set env: GMAIL_APP_PASSWORD
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "ngochieuvan123@gmail.com")


# ================== HELPERS ==================
def send_email_unlock(owner_name: str):
    if not APP_PASSWORD:
        raise RuntimeError("Chưa set GMAIL_APP_PASSWORD trong biến môi trường!")

    subject = "THONG BAO: DA MO KHOA"
    body = f"He thong vua mo khoa thanh cong cho chu so huu: {owner_name}"
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)


def preprocess_face(gray_face: np.ndarray, clahe) -> np.ndarray:
    """Chuẩn hoá ảnh mặt để LBPH ổn định hơn (ánh sáng, nhiễu)."""
    face = cv.resize(gray_face, IMG_SIZE)
    face = clahe.apply(face)
    face = cv.GaussianBlur(face, (3, 3), 0)
    return face


def pick_best_face(faces):
    """
    Chọn face ưu tiên:
    - diện tích lớn hơn (gần camera hơn) là ưu tiên
    - nếu bằng nhau thì cứ lấy cái đầu
    """
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    return faces[0]


# ================== MAIN ==================
def main():
    # Load model
    recog = cv.face.LBPHFaceRecognizer_create()
    recog.read("face_recognizer_model.yml")
    label_dict = np.load("label_dict.npy", allow_pickle=True).item()

    # Face cascade (alt2 thường ổn hơn default)
    cascade_path = cv.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    face_cascade = cv.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Không load được cascade:", cascade_path)
        return

    # Preprocess tool
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi camera!")
        return

    # Voting buffer
    hits = deque(maxlen=WINDOW_FRAMES)
    last_unlock_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # (Optional) lật ngang cho giống gương
        frame = cv.flip(frame, 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

        best_name = "Unknown"
        best_conf = 9999.0
        best_rect = None

        # Nếu có nhiều mặt, lấy mặt to nhất (thường là chủ thể)
        rect = pick_best_face(faces)
        if rect is not None:
            (x, y, w, h) = rect
            best_rect = rect

            face_roi = gray[y:y + h, x:x + w]
            face_roi = preprocess_face(face_roi, clahe)

            pred_label, conf = recog.predict(face_roi)
            name = label_dict.get(pred_label, "Unknown")

            best_name = name
            best_conf = float(conf)

        # Quyết định match
        is_owner = (best_name == OWNER_NAME and best_conf <= THRESHOLD)
        hits.append(1 if is_owner else 0)

        unlocked = (sum(hits) >= REQUIRED_HITS)

        # ===== UI draw =====
        if best_rect is not None:
            (x, y, w, h) = best_rect
            display_name = best_name if best_conf <= THRESHOLD else "Unknown"
            color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)

            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(
                frame,
                f"{display_name} ({best_conf:.1f})",
                (x, max(30, y - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
        else:
            # không thấy mặt
            display_name = "No face"
            best_conf = 9999.0

        # Status text
        if unlocked:
            cv.putText(frame, "DA MO KHOA", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            now = time.time()
            if now - last_unlock_time >= UNLOCK_COOLDOWN_SEC:
                last_unlock_time = now
                print("[UNLOCK] SUCCESS -> sending email...")
                try:
                    send_email_unlock(OWNER_NAME)
                    print("[EMAIL] Sent!")
                except Exception as e:
                    print("[EMAIL] Failed:", e)
        else:
            cv.putText(frame, "CHUA MO KHOA", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Debug vote bar
        cv.putText(frame, f"Vote: {sum(hits)}/{WINDOW_FRAMES}  thr={THRESHOLD}",
                   (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow("Face Unlock (LBPH Improved)", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()