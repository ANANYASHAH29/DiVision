"""Fine-tune YOLO11-Seg on CarDD and install the best weights where the API expects them.

Usage:  python scripts/train.py --epochs 100 --model yolo11s-seg.pt --device 0
(CPU works but is very slow; use a GPU or Google Colab / Kaggle for the real run.)
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/cardd_yolo/cardd.yaml")
ap.add_argument("--model", default="yolo11s-seg.pt")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project="data/runs", name="damage_seg", exist_ok=True)
best = Path("data/runs/damage_seg/weights/best.pt")
dst = Path("data/weights/damage_seg.pt")
shutil.copy(best, dst)
print("Installed", dst)
