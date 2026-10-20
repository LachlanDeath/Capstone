"""Train the cello keypoint model on your GPU.

    python train_cello.py              # yolo11n-pose (fastest)
    python train_cello.py --model yolo11s-pose.pt   # a bit more accurate, still fast on an RTX 4060

Result: runs/pose/cello/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--data", default="dataset_yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")          # "0" = your GPU, "cpu" = no GPU
    a = ap.parse_args()

    model = YOLO(a.model)                             # downloads the pretrained weights the first time
    model.train(
        data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, device=a.device,
        patience=40,                                  # stop early if validation stops improving
        workers=2,                                    # keep low on Windows
        project="runs/pose", name="cello", exist_ok=True,
        fliplr=0.5,                                   # mirror augmentation (you mirror frames at runtime)
        degrees=8, translate=0.1, scale=0.35,         # small rotations / shifts / zoom
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,            # lighting variation
        mosaic=1.0, close_mosaic=15,
    )
    print("Best weights: runs/pose/cello/weights/best.pt")


if __name__ == "__main__":                            # required on Windows
    main()