import argparse
import depth_pro
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(description="Depth Pro CLI Aracı")
    parser.add_argument("-i", "--image", required=True, help="Giriş resim yolu")
    parser.add_argument(
        "-o", "--output", required=True, help="Çıkış derinlik haritası yolu"
    )
    args = parser.parse_args()

    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image, _, f_px = depth_pro.load_rgb(args.image)
    image = transform(image)

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]

    depth_np = depth.squeeze().cpu().numpy()
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255.0
    depth_np = depth_np.astype(np.uint8)
    cv2.imwrite(args.output, depth_np)
    print(f"Derinlik haritası {args.output} yoluna kaydedildi.")


if __name__ == "__main__":
    main()
