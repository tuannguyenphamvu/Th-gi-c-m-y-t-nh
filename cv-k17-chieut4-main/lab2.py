import cv2 as cv
import numpy as np
import math

# =========================
# CẤU HÌNH
# =========================
VIDEO_PATH = "bang_chuyen (1).mp4"

LINE_X = 600          # vị trí line đỏ (tọa độ x). Bạn chỉnh cho đúng vị trí trong video
LINE_COLOR = (0, 0, 255)
LINE_THICK = 3

# Đếm hướng: "LR" = Left->Right, "RL" = Right->Left, "BOTH" = cả hai
COUNT_DIR = "LR"

# Tracking đơn giản
MAX_DIST = 60         # khoảng cách tối đa để ghép cùng 1 ID giữa 2 frame
MAX_MISSES = 10       # số frame mất dấu trước khi xóa track

# Tốc độ phát
speed = 1.0           # 1.0 = bình thường; 2.0 = nhanh gấp đôi; 0.5 = chậm nửa

# =========================
# HÀM PHỤ
# =========================
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =========================
# TRACKS DATA
# =========================
# tracks[id] = {
#   "center": (x, y),
#   "prev_x": x_prev,
#   "miss": int,
#   "counted": bool
# }
tracks = {}
next_id = 0
count = 0

# =========================
# OPEN VIDEO
# =========================
vid = cv.VideoCapture(VIDEO_PATH)
if not vid.isOpened():
    raise FileNotFoundError(f"Không mở được video: {VIDEO_PATH}")

fps = vid.get(cv.CAP_PROP_FPS)
if fps <= 1e-6:
    fps = 25.0
base_delay = int(1000 / fps)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    # nếu LINE_X vượt khung hình thì kéo về hợp lệ
    LINE_X = max(0, min(LINE_X, w - 1))

    # =========================
    # 1) TIỀN XỬ LÝ + TÌM CIRCLE
    # =========================
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 2)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=120,    # edge threshold
        param2=25,     # càng nhỏ càng dễ ra nhiều circle (nhiều nhiễu hơn)
        minRadius=8,
        maxRadius=80
    )

    detections = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            detections.append((int(x), int(y), int(r)))

    # =========================
    # 2) GHÉP DETECTION VÀO TRACK (NEAREST NEIGHBOR)
    # =========================
    used_det = set()
    updated_ids = set()

    # ghép từng track cũ với detection gần nhất
    for tid, tinfo in list(tracks.items()):
        best_j = -1
        best_d = 1e9
        tx, ty = tinfo["center"]

        for j, (x, y, r) in enumerate(detections):
            if j in used_det:
                continue
            d = dist((tx, ty), (x, y))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j != -1 and best_d <= MAX_DIST:
            # update track
            x, y, r = detections[best_j]
            used_det.add(best_j)
            updated_ids.add(tid)

            prev_x = tinfo["center"][0]
            tracks[tid]["center"] = (x, y)
            tracks[tid]["prev_x"] = prev_x
            tracks[tid]["miss"] = 0
            tracks[tid]["r"] = r
        else:
            # không ghép được -> tăng miss
            tracks[tid]["miss"] += 1

    # track nào miss quá lâu thì xóa
    for tid in list(tracks.keys()):
        if tracks[tid]["miss"] > MAX_MISSES:
            del tracks[tid]

    # detection còn lại -> tạo track mới
    for j, (x, y, r) in enumerate(detections):
        if j in used_det:
            continue
        tracks[next_id] = {
            "center": (x, y),
            "prev_x": x,
            "miss": 0,
            "counted": False,
            "r": r
        }
        next_id += 1

    # =========================
    # 3) ĐẾM KHI VƯỢT LINE
    # =========================
    for tid, tinfo in tracks.items():
        x, y = tinfo["center"]
        prev_x = tinfo["prev_x"]

        crossed_LR = (prev_x < LINE_X and x >= LINE_X)
        crossed_RL = (prev_x > LINE_X and x <= LINE_X)

        do_count = False
        if COUNT_DIR == "LR" and crossed_LR:
            do_count = True
        elif COUNT_DIR == "RL" and crossed_RL:
            do_count = True
        elif COUNT_DIR == "BOTH" and (crossed_LR or crossed_RL):
            do_count = True

        if do_count and (not tinfo["counted"]):
            count += 1
            tracks[tid]["counted"] = True

    # =========================
    # 4) VẼ UI
    # =========================
    # vẽ line đỏ
    cv.line(frame, (LINE_X, 0), (LINE_X, h), LINE_COLOR, LINE_THICK)
    cv.putText(frame, "LINE 1", (LINE_X + 10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, LINE_COLOR, 2)

    # vẽ circle + id
    for tid, tinfo in tracks.items():
        x, y = tinfo["center"]
        r = int(tinfo.get("r", 10))
        cv.circle(frame, (x, y), r, (255, 255, 255), 2)
        cv.circle(frame, (x, y), 2, (0, 255, 255), -1)
        cv.putText(frame, f"ID:{tid}", (x - 25, y - r - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.putText(frame, f"Count: {count}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Speed: {speed:.2f}   (+/- to change)", (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv.imshow("video", frame)

    # delay theo speed
    delay = max(1, int(base_delay / speed))

    key = cv.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("+") or key == ord("="):
        speed = min(10.0, speed + 0.25)
    elif key == ord("-") or key == ord("_"):
        speed = max(0.1, speed - 0.25)
    elif key == ord("r"):
        # reset count
        count = 0
        for tid in tracks:
            tracks[tid]["counted"] = False

vid.release()
cv.destroyAllWindows()
print("Tong so hinh tron vuot line:", count)
