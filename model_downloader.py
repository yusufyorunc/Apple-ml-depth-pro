import os
import urllib.request
import torch

MODEL_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
MODEL_PATH = "checkpoints/depth_pro.pt"


def download_model():
    if os.path.exists(MODEL_PATH):
        try:
            torch.load(MODEL_PATH)
            print("Model başarıyla yüklendi. Tekrar indirmeye gerek yok.")
            return
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            print("Model siliniyor ve tekrar indiriliyor...")
            os.remove(MODEL_PATH)

    print("Model indiriliyor...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model başarıyla indirildi.")


if __name__ == "__main__":
    download_model()
