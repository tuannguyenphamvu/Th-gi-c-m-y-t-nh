import os
import urllib.request

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

files = {
    "deploy.prototxt": "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
}

def download(url: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[SKIP] {out_path} (exists)")
        return
    print(f"[DOWN] {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[OK]  {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

def main():
    for name, url in files.items():
        out_path = os.path.join(MODEL_DIR, name)
        download(url, out_path)

    print("\nDone. You should have:")
    for name in files.keys():
        print(" -", os.path.join(MODEL_DIR, name))

if __name__ == "__main__":
    main()