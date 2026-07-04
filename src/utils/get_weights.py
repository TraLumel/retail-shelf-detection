import shutil
import socket
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "weights"

HF_SOURCES = {
    "yolov8n.pt": ("Ultralytics/YOLOv8", "yolov8n.pt"),
    "yolo11n.pt": ("Ultralytics/YOLO11", "yolo11n.pt"),
}
RTDETR_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt"


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)

    from huggingface_hub import hf_hub_download
    for fname, (repo, fn) in HF_SOURCES.items():
        dst = WEIGHTS_DIR / fname
        if dst.exists():
            print(f"{fname}: уже скачан")
            continue
        print(f"{fname}: скачиваю с HuggingFace ({repo})...")
        cached = hf_hub_download(repo_id=repo, filename=fn)
        shutil.copy(cached, dst)
        print(f"{fname}: OK ({dst.stat().st_size / 1e6:.1f} MB)")

    dst = WEIGHTS_DIR / "rtdetr-l.pt"
    if dst.exists():
        print("rtdetr-l.pt: уже скачан")
        return
    print("rtdetr-l.pt: пробую GitHub (таймаут 30 сек)...")
    socket.setdefaulttimeout(30)
    try:
        urllib.request.urlretrieve(RTDETR_URL, dst)
        print(f"rtdetr-l.pt: OK ({dst.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        if dst.exists():
            dst.unlink()
        print(f"rtdetr-l.pt: не скачался ({type(e).__name__}). "
              "Не страшно: вместо RT-DETR будет обучена RetinaNet.")


if __name__ == "__main__":
    main()
