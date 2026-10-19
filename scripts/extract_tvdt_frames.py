"""Extract frames from TVDT walk-around videos for tracking evaluation (~1,700 frames/video).

Usage:  python scripts/extract_tvdt_frames.py --src data/raw/TVDT --out data/tvdt_frames --every 5
"""
import argparse
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, type=Path)
ap.add_argument("--out", type=Path, default=Path("data/tvdt_frames"))
ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()

total = 0
for v in sorted([*a.src.rglob("*.mp4"), *a.src.rglob("*.avi"), *a.src.rglob("*.mov")]):
    d = a.out / v.stem
    d.mkdir(parents=True, exist_ok=True)
    cap, i, n = cv2.VideoCapture(str(v)), 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            cv2.imwrite(str(d / f"{i:05d}.jpg"), f)
            n += 1
        i += 1
    print(v.name, n, "frames")
    total += n
print("total", total)
