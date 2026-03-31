import cv2 as cv
import numpy as np
import urllib.request
from urllib.error import URLError

def read_img_url(url):
    try:
        req = urllib.request.urlopen(url, timeout=10)
        img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv.imdecode(img_rw, cv.IMREAD_COLOR)
        # Resize lại một chút cho dễ nhìn khi ghép 3 ảnh
        img = cv.resize(img, (400, 300))
        return img
    except URLError:
        print("Lỗi: Không thể tải ảnh từ URL.")
        return None

def add_salt_and_pepper_noise(img, amount=0.02):
    """Thêm nhiễu muối tiêu vào ảnh"""
    noisy = img.copy()
    # Muối (trắng)
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 255

    # Tiêu (đen)
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

def clean_noise(img):
    """Làm sạch nhiễu bằng Median Blur"""
    # ksize=5 là độ mạnh của bộ lọc, phải là số lẻ
    return cv.medianBlur(img, 5)

# ...existing code...
if __name__=="__main__":
    img_path = r"c:\Nhap\New folder\z7476545799116_c521856c8a735b51ec6d79b007da8fcf.jpg"
    original = cv.imread(img_path)
    
    if original is not None:
        # 1. Tạo ảnh nhiễu
        noisy_img = add_salt_and_pepper_noise(original.copy(), amount=0.05)
        
        # 2. Làm sạch ảnh nhiễu
        cleaned_img = clean_noise(noisy_img.copy())
        
        # 3. Tạo ảnh đường nét
        gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 200)

        # HIỂN THỊ TÁCH RỜI TỪNG CỬA SỔ
        cv.imshow("1. Anh Goc", original)
        cv.imshow("2. Anh Bi Nhieu", noisy_img)
        cv.imshow("3. Anh Da Lam Sach", cleaned_img)
        cv.imshow("4. Duong Net (Canny)", edges)

        print("Bam phím bat ky tren ban phim de dong tat ca cua so...")
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Khong tim thay file anh. Vui long kiem tra lai duong dan.")