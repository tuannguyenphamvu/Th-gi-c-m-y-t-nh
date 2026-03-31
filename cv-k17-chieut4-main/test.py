import cv2 as cv
import numpy as np
import urllib.request

def read_img_url(url):
    req = urllib.request.urlopen(url)
    img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
    return cv.imdecode(img_rw, cv.IMREAD_COLOR)

def region_of_interest(bin_img):
    h, w = bin_img.shape[:2]
    mask = np.zeros_like(bin_img)

    # ROI rộng hơn để bắt cả cong + nhiều làn
    poly = np.array([[
        (int(0.05*w), h),
        (int(0.43*w), int(0.62*h)),
        (int(0.57*w), int(0.62*h)),
        (int(0.95*w), h)
    ]], dtype=np.int32)

    cv.fillPoly(mask, poly, 255)
    return cv.bitwise_and(bin_img, mask)

def color_mask_lane(bgr):
    # HLS rất hợp cho lane detection (white/yellow)
    hls = cv.cvtColor(bgr, cv.COLOR_BGR2HLS)

    # White lane: Lightness cao
    white_lower = np.array([0, 200, 0], dtype=np.uint8)
    white_upper = np.array([180, 255, 255], dtype=np.uint8)
    white = cv.inRange(hls, white_lower, white_upper)

    # Yellow lane: H khoảng vàng, S vừa/cao, L vừa
    yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
    yellow = cv.inRange(hls, yellow_lower, yellow_upper)

    mask = cv.bitwise_or(white, yellow)

    # Làm sạch mask (giảm nhiễu do xe/texture)
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return mask

def enhance_contrast_gray(bgr):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def hough_lines(bin_img):
    return cv.HoughLinesP(
        bin_img,
        rho=1,
        theta=np.pi/180,
        threshold=35,
        minLineLength=40,
        maxLineGap=180
    )

def average_lane_lines(img, lines):
    if lines is None:
        return None, None

    h, w = img.shape[:2]
    left, right = [], []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x1 == x2:
            continue

        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 0.35:   # bỏ line gần ngang (rất hay là nhiễu / xe)
            continue

        b = y1 - m*x1

        # lọc theo vị trí: trái nằm nửa trái, phải nằm nửa phải
        x_mid = (x1 + x2) / 2
        if m < 0 and x_mid < w * 0.55:
            left.append((m, b))
        elif m > 0 and x_mid > w * 0.45:
            right.append((m, b))

    def make_points(mb):
        m, b = mb
        y1 = h
        y2 = int(h * 0.62)
        if abs(m) < 1e-6:
            return None
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        return (x1, y1), (x2, y2)

    left_line = make_points(np.mean(left, axis=0)) if len(left) else None
    right_line = make_points(np.mean(right, axis=0)) if len(right) else None
    return left_line, right_line

def detect_lanes(img):
    # 1) tăng tương phản cho gradient (đỡ bị xe/đổ bóng)
    gray = enhance_contrast_gray(img)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # 2) edge
    edges = cv.Canny(blur, 60, 160)

    # 3) mask màu lane (white + yellow)
    c_mask = color_mask_lane(img)

    # 4) combine: (edges AND color_mask) -> bám vạch tốt hơn
    combo = cv.bitwise_and(edges, c_mask)

    # 5) ROI
    roi = region_of_interest(combo)

    # 6) Hough + gộp line
    lines = hough_lines(roi)
    left_line, right_line = average_lane_lines(img, lines)

    # 7) vẽ kết quả
    line_layer = np.zeros_like(img)

    if left_line is not None:
        cv.line(line_layer, left_line[0], left_line[1], (0,255,0), 10)
    if right_line is not None:
        cv.line(line_layer, right_line[0], right_line[1], (0,255,0), 10)

    # (tuỳ chọn) tô vùng lane
    if left_line is not None and right_line is not None:
        pts = np.array([
            left_line[0], left_line[1], right_line[1], right_line[0]
        ], dtype=np.int32)
        cv.fillPoly(line_layer, [pts], (0,255,0))

    out = cv.addWeighted(img, 0.85, line_layer, 0.6, 0)

    return out, c_mask, edges, combo, roi

if __name__ == "__main__":
    url = "file:///C:/Users/Chieu/Downloads/z7476545799116_c521856c8a735b51ec6d79b007da8fcf.jpg"
    img = read_img_url(url)

    out, c_mask, edges, combo, roi = detect_lanes(img)

    cv.imshow("Original", img)
    cv.imshow("Color mask (white+yellow)", c_mask)
    cv.imshow("Edges", edges)
    cv.imshow("Edges AND ColorMask", combo)
    cv.imshow("ROI", roi)
    cv.imshow("Lane Detected (stable)", out)
    cv.waitKey(0)
    cv.destroyAllWindows()
