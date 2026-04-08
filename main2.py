import cv2
import easyocr
import numpy as np

# --- Cấu hình các tham số ---
# Đường dẫn video của bạn
video_path = 'plate2.mp4' 
# Tọa độ y của vạch kẻ giữa màn hình (giả sử là 350)
line_y = 350 
# Diện tích tối thiểu để được coi là một chiếc xe (điều chỉnh cho phù hợp)
min_area = 4000 
# Chiều cao của vùng crop biển số (so với chiều cao xe, vd: 1/3)
plate_crop_ratio = 0.3
# --- End Cấu hình ---

# 1. Khởi tạo EasyOCR (load model đọc chữ, chỉ cần thực hiện 1 lần)
reader = easyocr.Reader(['en'])

# 2. Khởi tạo bộ lọc tách nền (MOG2) - thử giảm varThreshold để nhạy hơn
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- TIỀN XỬ LÝ ẢNH ---
    # 3. Lọc tách nền
    fgMask = backSub.apply(frame)
    
    # 4. Lọc nhiễu
    # Giảm threshold để giữ lại nhiều chi tiết chuyển động hơn (vd: 200 thay vì 250)
    _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
    # Áp dụng xói mòn và giãn nở để lấp đầy các lỗ trống trong contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)

    # --- PHÁT HIỆN XE (CHUYỂN ĐỘNG) ---
    # 5. Tìm các đường bao (Contours)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vẽ line vàng mốc
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

    detected_vehicles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        
        # --- LOGIC KIỂM TRA CROSSING ---
        # 6. Kiểm tra nếu MÉP DƯỚI của xe đã VƯỢT QUA vạch
        if (y + h) > line_y:
            # Highlight line khi có xe vượt qua
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 5)
            
            # Lưu thông tin xe đã detect để xử lý OCR sau (tránh duplicate vẽ)
            detected_vehicles.append((x, y, w, h))

    # --- NHẬN DIỆN BIỂN SỐ VÀ VẼ LABEL 'BXS' ---
    for (x, y, w, h) in detected_vehicles:
        # Vẽ BBox cho xe (Màu vàng theo yêu cầu)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        # 7. Cắt vùng ảnh chứa biển số (Roi)
        # Giả sử biển số nằm ở khoảng nửa dưới của xe
        roi_y = int(y + h * (1 - plate_crop_ratio))
        roi_h = int(h * plate_crop_ratio)
        roi = frame[roi_y : roi_y + roi_h, x : x + w]
        if roi.size > 0:
            # --- Code EasyOCR ---
            result = reader.readtext(roi)
            
            for (bbox_ocr, text, prob) in result:
                # Chỉ lấy các chữ có độ tin cậy tương đối (vd > 0.3)
                if prob > 0.3:
                    # 8. Vẽ label 'BXS' và biển số lên khung hình
                    label_text = f"BXS: {text.upper()}"
                    cv2.putText(frame, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"Phát hiện biển số tại ({x},{y}): {text} - Độ tin cậy: {prob}")

    cv2.imshow('Vehicle Detection & BXS', frame)
    # cv2.imshow('Mask', fgMask) # Xem mặt nạ chuyển động

    # Nhấn 'q' để thoát
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()