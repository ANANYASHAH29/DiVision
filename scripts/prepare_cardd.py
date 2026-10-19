"""Convert CarDD (COCO json) into the YOLO-seg layout ultralytics trains on.

Expected input (as released by the CarDD authors):
    <src>/CarDD_COCO/annotations/instances_{train,val,test}2017.json
    <src>/CarDD_COCO/{train,val,test}2017/*.jpg
Usage:  python scripts/prepare_cardd.py --src data/raw/CarDD_release
"""
import argparse
import json
import shutil
from pathlib import Path

NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]


def polys_from(seg, h, w):
    if isinstance(seg, list):
        return [p for p in seg if len(p) >= 6]
    try:  # RLE -> polygons
        import cv2
        import numpy as np
        from pycocotools import mask as mu
        rle = mu.frPyObjects(seg, h, w) if isinstance(seg["counts"], list) else seg
        m = mu.decode(rle).astype("uint8")
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1).tolist() for c in cs if len(c) >= 3]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/cardd_yolo"))
    a = ap.parse_args()
    root = next(a.src.rglob("annotations")).parent
    for split in ("train", "val", "test"):
        ann = root / "annotations" / f"instances_{split}2017.json"
        if not ann.exists():
            print("skip", split)
            continue
        coco = json.loads(ann.read_text())
        cats = {c["id"]: NAMES.index(c["name"].lower().replace("_", " ")) for c in coco["categories"]}
        imgs = {i["id"]: i for i in coco["images"]}
        lines: dict[int, list[str]] = {i: [] for i in imgs}
        for an in coco["annotations"]:
            im = imgs[an["image_id"]]
            for p in polys_from(an["segmentation"], im["height"], im["width"]):
                xy = [f"{v / (im['width'] if k % 2 == 0 else im['height']):.6f}" for k, v in enumerate(p)]
                lines[an["image_id"]].append(f"{cats[an['category_id']]} " + " ".join(xy))
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for iid, im in imgs.items():
            src = root / f"{split}2017" / im["file_name"]
            if not src.exists():
                continue
            shutil.copy(src, a.out / "images" / split / src.name)
            (a.out / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines[iid]))
        print(split, len(imgs), "images")
    (a.out / "cardd.yaml").write_text(
        f"path: {a.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print("wrote", a.out / "cardd.yaml")


if __name__ == "__main__":
    main()
