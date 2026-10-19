"""Stage 1 of the damage verifier: run the detector, label every detection, cache crop embeddings.

Sources
  cardd_val / cardd_test : real labels. A detection is "real" (1) if its box overlaps ANY human-labelled
                           damage box (IoU >= 0.3, class-agnostic), otherwise a false alarm (0).
  eval_cleaned / eval_reflections : photos of undamaged cars, so every detection is a false alarm (0).
  eval_wsglass : real broken-windshield photos, only used to check real damage is not suppressed (-1).

Output: data/verifier/mined.npz  (+ mined_meta.json)
Usage:  python scripts/verifier_mine.py
"""
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "verifier"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
CONF = 0.25


def iou_any(box, gts):
    if not len(gts):
        return 0.0
    g = np.asarray(gts, float)
    x1, y1 = np.maximum(box[0], g[:, 0]), np.maximum(box[1], g[:, 1])
    x2, y2 = np.minimum(box[2], g[:, 2]), np.minimum(box[3], g[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return float((inter / (a + b - inter + 1e-9)).max())


def crop_of(img, box, pad=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    return img[y1:y2, x1:x2]


def tabular(cls_id, conf, box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    onehot = np.eye(len(CLASSES))[cls_id]
    area = (x2 - x1) * (y2 - y1) / (w * h)
    return np.concatenate([onehot, [conf, np.sqrt(area), np.log((x2 - x1 + 1) / (y2 - y1 + 1)),
                                    (x1 + x2) / 2 / w, (y1 + y2) / 2 / h]])


def sources():
    ann = ROOT / "data" / "cardd_subset" / "annotations"
    for split in ("val", "test"):
        coco = json.loads((ann / f"instances_{split}2017.json").read_text())
        gts = {}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            gts.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
        files = {}
        for im in coco["images"]:
            files[str(ROOT / "data" / "cardd_subset" / f"{split}2017" / im["file_name"])] = gts.get(im["id"], [])
        yield f"cardd_{split}", files
    ev = ROOT / "data" / "eval"
    for name, label in (("cleaned", 0), ("reflections", 0), ("wsglass_broken", -1)):
        fs = sorted(glob.glob(str(ev / name / "*.jp*g")))
        yield f"eval_{name.replace('_broken', '')}", {f: label for f in fs}


def main():
    det = YOLO(str(ROOT / "data" / "weights" / "damage_seg.pt"))
    emb = YOLO("yolo11n-cls.pt")
    rec = dict(emb=[], tab=[], y=[], src=[], img=[], conf=[], cls=[])
    meta = {}
    for src, files in sources():
        paths = sorted(files)
        meta[src] = len(paths)
        print(src, len(paths), "images", flush=True)
        for i in range(0, len(paths), 8):
            batch = paths[i:i + 8]
            for p, r in zip(batch, det.predict(batch, conf=CONF, imgsz=640, verbose=False)):
                if not len(r.boxes):
                    continue
                img = r.orig_img
                crops, boxes = [], r.boxes.xyxy.cpu().numpy()
                keep = [k for k, b in enumerate(boxes) if crop_of(img, b).size]
                crops = [crop_of(img, boxes[k]) for k in keep]
                if not crops:
                    continue
                e = emb.embed(crops, imgsz=224, verbose=False)
                for k, ev in zip(keep, e):
                    c, cf = int(r.boxes.cls[k]), float(r.boxes.conf[k])
                    gt = files[p]
                    y = (1 if iou_any(boxes[k], gt) >= 0.3 else 0) if isinstance(gt, list) else gt
                    rec["emb"].append(np.asarray(ev.cpu()).reshape(-1))
                    rec["tab"].append(tabular(c, cf, boxes[k], img.shape))
                    rec["y"].append(y)
                    rec["src"].append(src)
                    rec["img"].append(f"{src}:{paths.index(p)}")
                    rec["conf"].append(cf)
                    rec["cls"].append(c)
            if (i // 8) % 20 == 0:
                print("  ", src, i, "/", len(paths), "dets so far", len(rec["y"]), flush=True)
    np.savez_compressed(OUT / "mined.npz", **{k: np.asarray(v) for k, v in rec.items()})
    (OUT / "mined_meta.json").write_text(json.dumps(meta))
    print("saved", len(rec["y"]), "detections", meta)


if __name__ == "__main__":
    main()
