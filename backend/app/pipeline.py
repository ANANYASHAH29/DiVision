"""Detect + segment (YOLO11-Seg) -> track (BoT-SORT) -> verify (reflection-aware) -> evidence records."""
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config, reflection, verifier
from .pipeline_exts import IMAGE_EXTS  # noqa: F401
from .schemas import TrackEvidence

_models = {}
MAX_EVIDENCE_IMAGES = 25
RED, ORANGE = (0, 0, 255), (0, 140, 255)


class WeightsMissing(RuntimeError):
    pass


def _load(path: Path):
    if not path.exists():
        raise WeightsMissing(f"Model weights not found: {path}. Train with scripts/train.py first.")
    if path not in _models:
        from ultralytics import YOLO  # imported lazily: heavy
        _models[path] = YOLO(str(path))
    return _models[path]


def _mask_centroid_inside(mask_xy: np.ndarray, part_masks: list[tuple[str, np.ndarray]]) -> Optional[str]:
    """Return the part whose polygon contains the damage centroid."""
    if mask_xy is None or len(mask_xy) == 0:
        return None
    cx, cy = mask_xy.mean(axis=0)
    for name, poly in part_masks:
        if cv2.pointPolygonTest(poly.astype(np.float32), (float(cx), float(cy)), False) >= 0:
            return name
    return None


def _parts_for_frame(parts_model, frame) -> list[tuple[str, np.ndarray]]:
    if parts_model is None:
        return []
    r = parts_model.predict(frame, conf=0.4, verbose=False)[0]
    if r.masks is None:
        return []
    return [(r.names[c], p) for c, p in zip(r.boxes.cls.int().tolist(), r.masks.xy) if len(p) >= 3]


def _annotate(frame, poly, box, label, color=RED):
    img = frame.copy()
    if poly is not None and len(poly) >= 3:
        p = poly.astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [p], color)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.polylines(img, [p], True, color, 2)
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def _mode(xs):
    return Counter(xs).most_common(1)[0][0] if xs else None


def _finish(tid, classes, confs, part, idx, best, claim_id, n_frames, first, write_image: bool,
            image_name: Optional[str] = None) -> TrackEvidence:
    """Verify one track on its best frame and build its evidence record."""
    thr = verifier.threshold() if verifier.available() else None
    p_real = verifier.score(best["frame"], best["box"], best["cls_id"], best["conf"])
    ra = reflection.assess(classes, idx, p_real, thr)
    label = f"#{tid} {_mode(classes)}" + (f" / {part}" if part else "") + f" {best['conf']:.2f}"
    if ra["likely"]:
        label += " [reflection?]"
    img_file = None
    if write_image:
        img_file = f"{claim_id}_{'p' if image_name else 't'}{tid}.jpg"
        cv2.imwrite(str(config.EVIDENCE / img_file),
                    _annotate(best["frame"], best["poly"], best["box"], label, ORANGE if ra["likely"] else RED))
    area = float(cv2.contourArea(best["poly"].astype(np.float32))) if best["poly"] is not None else 0.0
    return TrackEvidence(
        track_id=tid, damage=_mode(classes), component=part,
        max_conf=round(max(confs), 3), mean_conf=round(float(np.mean(confs)), 3),
        n_frames=n_frames, first_frame=first, best_frame=best["frame_idx"], mask_area_px=round(area, 1),
        evidence_image=img_file, image_name=image_name, reflection_risk=ra["risk"],
        likely_reflection=ra["likely"], flags=ra["flags"],
        real_prob=None if p_real is None else round(float(p_real), 3))


def run(video_path: Path, claim_id: str) -> tuple[list[TrackEvidence], int]:
    damage_model = _load(config.DAMAGE_WEIGHTS)
    parts_model = _load(config.PARTS_WEIGHTS) if config.PARTS_WEIGHTS.exists() else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracks: dict[int, dict] = defaultdict(lambda: dict(confs=[], classes=[], parts=[], idx=[], first=None, best=None))
    frame_idx = analyzed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % config.STRIDE == 0:
            analyzed += 1
            # persist=True keeps BoT-SORT state (Kalman + appearance ReID) across calls
            res = damage_model.track(frame, persist=True, tracker="botsort.yaml",
                                     conf=config.CONF_THRESHOLD, verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None:
                parts = None
                for i, tid in enumerate(res.boxes.id.int().tolist()):
                    t = tracks[tid]
                    conf = float(res.boxes.conf[i])
                    cls_id = int(res.boxes.cls[i])
                    poly = res.masks.xy[i] if res.masks is not None else None
                    if parts is None:
                        parts = _parts_for_frame(parts_model, frame)
                    part = _mask_centroid_inside(poly, parts) if poly is not None else None
                    t["confs"].append(conf)
                    t["classes"].append(res.names[cls_id])
                    t["idx"].append(analyzed)
                    if part:
                        t["parts"].append(part)
                    t["first"] = frame_idx if t["first"] is None else t["first"]
                    if t["best"] is None or conf > t["best"]["conf"]:
                        # keep a reference to the frame; annotation is deferred until the track is final
                        t["best"] = dict(conf=conf, frame_idx=frame_idx, frame=frame, poly=poly,
                                         box=res.boxes.xyxy[i].cpu().numpy(), cls_id=cls_id)
        frame_idx += 1
    cap.release()
    damage_model.predictor = None  # reset tracker state for the next video

    out: list[TrackEvidence] = []
    ranked = sorted((tid for tid, t in tracks.items() if t["confs"]), key=lambda k: -max(tracks[k]["confs"]))
    written = 0
    for tid in ranked:
        t = tracks[tid]
        solid = len(t["confs"]) >= config.MIN_TRACK_FRAMES
        write = solid and written < MAX_EVIDENCE_IMAGES
        written += write
        out.append(_finish(tid, t["classes"], t["confs"], _mode(t["parts"]), t["idx"], t["best"], claim_id,
                           len(t["confs"]), t["first"], write))
        t["best"] = None  # free the frame
    return sorted(out, key=lambda e: -e.max_conf), analyzed


def run_images(folder: Path, claim_id: str) -> tuple[list[TrackEvidence], int]:
    """Still-photo path: detect + segment each photo independently (no tracking possible)."""
    model = _load(config.DAMAGE_WEIGHTS)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    out: list[TrackEvidence] = []
    n = 0
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        n += 1
        res = model.predict(img, conf=config.CONF_THRESHOLD, verbose=False)[0]
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            best = dict(conf=conf, frame_idx=0, frame=img, cls_id=cls_id, box=res.boxes.xyxy[i].cpu().numpy(),
                        poly=res.masks.xy[i] if res.masks is not None else None)
            out.append(_finish(len(out) + 1, [res.names[cls_id]], [conf], None, None, best, claim_id, 1, 0,
                               len(out) < MAX_EVIDENCE_IMAGES, image_name=path.name))
    return sorted(out, key=lambda e: -e.max_conf), n
