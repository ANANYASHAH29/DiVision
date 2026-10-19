"""Second-stage damage verifier: is this detection real damage, or a reflection / glare / droplets?

Trained by scripts/verifier_train.py on detector false alarms vs real damage. At run time it embeds the
detection's crop with a small ImageNet classifier and applies a logistic-regression head. The feature code
below must stay identical to scripts/verifier_mine.py.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

CLASSES = config.DAMAGE_CLASSES
WEIGHTS = config.DATA / "weights" / "verifier.npz"
EMBED_WEIGHTS = config.DATA / "weights" / "yolo11n-cls.pt"

_state: dict = {}


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


def available() -> bool:
    return WEIGHTS.exists()


def _load():
    if not _state:
        from ultralytics import YOLO
        z = np.load(WEIGHTS)
        emb_src = str(EMBED_WEIGHTS) if EMBED_WEIGHTS.exists() else "yolo11n-cls.pt"
        _state.update(theta=z["theta"], mu=z["mu"], sd=z["sd"], thr=float(z["threshold"]), emb=YOLO(emb_src))
    return _state


def threshold() -> float:
    return _load()["thr"]


def score(frame: np.ndarray, box, cls_id: int, conf: float) -> Optional[float]:
    """Probability (0..1) that the detection is real damage. None when no verifier is installed."""
    if not available():
        return None
    s = _load()
    crop = crop_of(frame, box)
    if crop.size == 0:
        return None
    e = s["emb"].embed([crop], imgsz=224, verbose=False)[0].cpu().numpy().reshape(-1)
    x = np.concatenate([e, tabular(cls_id, conf, box, frame.shape)]).astype(np.float32)
    z = float(((x - s["mu"]) / s["sd"]) @ s["theta"][:-1] + s["theta"][-1])
    return 1.0 / (1.0 + np.exp(-z))
