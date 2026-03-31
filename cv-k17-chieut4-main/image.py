import cv2 as cv
import numpy as np
import math
import datetime
import time

# =========================
# CẤU HÌNH
# =========================
W, H = 900, 900
cx, cy = W // 2, H // 2
R = 360  # bán kính mặt đồng hồ
BG_PURPLE = (180, 70, 200)  # tím (BGR)

# Màu kim
COLOR_HOUR = (255, 0, 0)    # xanh dương (BGR)
COLOR_MIN  = (0, 255, 0)    # xanh lá
COLOR_SEC  = (0, 0, 255)    # đỏ

# Độ dài kim
LEN_HOUR = int(R * 0.55)
LEN_MIN  = int(R * 0.75)
LEN_SEC  = int(R * 0.90)

# Độ dày kim
THICK_HOUR = 10
THICK_MIN  = 6
THICK_SEC  = 2

# Số La Mã
roman = ["XII", "I", "II", "III", "IV", "V",
         "VI", "VII", "VIII", "IX", "X", "XI"]

roman_colors = [
    (0, 255, 255),  # vàng
    (255, 0, 255),  # tím hồng
    (255, 255, 0),  # cyan
    (0, 165, 255),  # cam
    (0, 255, 0),    # xanh lá
    (255, 0, 0),    # xanh dương
    (0, 0, 255),    # đỏ
    (128, 0, 128),  # tím đậm
    (0, 128, 255),  # cam nhạt
    (255, 255, 255),# trắng
    (200, 200, 200),# xám
    (50, 200, 50),  # xanh
]

font = cv.FONT_HERSHEY_SIMPLEX


def angle_to_point(angle_rad, length):
    """Góc 0 rad hướng lên trên. Quay theo chiều kim đồng hồ."""
    x = int(cx + length * math.sin(angle_rad))
    y = int(cy - length * math.cos(angle_rad))
    return (x, y)


def draw_clock_face(img):
    # nền tím toàn ảnh
    img[:] = BG_PURPLE

    # vòng tròn mặt đồng hồ
    cv.circle(img, (cx, cy), R, (255, 255, 255), 8)
    cv.circle(img, (cx, cy), R - 6, (60, 60, 60), 2)

    # Level 5: vạch chỉ phút
    for i in range(60):
        ang = (2 * math.pi) * (i / 60.0)
        # vạch dài cho giờ, ngắn cho phút
        if i % 5 == 0:
            inner = R - 40
            thick = 6
            color = (255, 255, 255)
        else:
            inner = R - 25
            thick = 2
            color = (230, 230, 230)

        p1 = angle_to_point(ang, inner)
        p2 = angle_to_point(ang, R - 8)
        cv.line(img, p1, p2, color, thick, cv.LINE_AA)

    # Số La Mã (12 vị trí)
    for k in range(12):
        ang = (2 * math.pi) * (k / 12.0)  # 0->XII ở trên
        pos = angle_to_point(ang, R - 90)

        text = roman[k]
        color = roman_colors[k % len(roman_colors)]

        # canh giữa chữ
        (tw, th), _ = cv.getTextSize(text, font, 1.5, 3)
        tx = pos[0] - tw // 2
        ty = pos[1] + th // 2

        # viền đen cho dễ nhìn
        cv.putText(img, text, (tx, ty), font, 1.5, (0, 0, 0), 6, cv.LINE_AA)
        cv.putText(img, text, (tx, ty), font, 1.5, color, 3, cv.LINE_AA)

    # tâm đồng hồ
    cv.circle(img, (cx, cy), 10, (255, 255, 255), -1)
    cv.circle(img, (cx, cy), 10, (0, 0, 0), 2)


def draw_hands(img, now):
    # Lấy thời gian thật
    h = now.hour % 12
    m = now.minute
    s = now.second
    micro = now.microsecond

    # Level 2: kim giây chuyển động mượt
    sec = s + micro / 1_000_000.0
    ang_sec = 2 * math.pi * (sec / 60.0)

    # Level 3: kim phút chuyển động theo giây
    minute = m + sec / 60.0
    ang_min = 2 * math.pi * (minute / 60.0)

    # Level 4: kim giờ chuyển động theo phút
    hour = h + minute / 60.0
    ang_hour = 2 * math.pi * (hour / 12.0)

    # Tính điểm cuối kim
    p_hour = angle_to_point(ang_hour, LEN_HOUR)
    p_min  = angle_to_point(ang_min, LEN_MIN)
    p_sec  = angle_to_point(ang_sec, LEN_SEC)

    # Vẽ kim (giờ, phút, giây)
    cv.line(img, (cx, cy), p_hour, COLOR_HOUR, THICK_HOUR, cv.LINE_AA)
    cv.line(img, (cx, cy), p_min,  COLOR_MIN,  THICK_MIN,  cv.LINE_AA)
    cv.line(img, (cx, cy), p_sec,  COLOR_SEC,  THICK_SEC,  cv.LINE_AA)

    # nút chốt đỏ ở tâm
    cv.circle(img, (cx, cy), 7, COLOR_SEC, -1)


def main():
    cv.namedWindow("Clock", cv.WINDOW_AUTOSIZE)

    # Vòng lặp cập nhật theo thời gian thật
    while True:
        img = np.zeros((H, W, 3), dtype=np.uint8)

        draw_clock_face(img)
        now = datetime.datetime.now()
        draw_hands(img, now)

        # hiển thị thời gian số (tuỳ thích)
        tstr = now.strftime("%H:%M:%S")
        cv.putText(img, tstr, (30, 60), font, 1.5, (0, 0, 0), 6, cv.LINE_AA)
        cv.putText(img, tstr, (30, 60), font, 1.5, (255, 255, 255), 3, cv.LINE_AA)

        cv.imshow("Clock", img)

        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        # Giữ fps ổn định (tuỳ máy). Bạn có thể bỏ.
        time.sleep(0.005)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
