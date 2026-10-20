"""Interior inspection: odometer reading, warning lights, interior condition."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np

from pydantic import BaseModel

from . import config


class InteriorReport(BaseModel):
    odometer_km: Optional[int] = None            # None = could not be read
    odometer_confidence: Optional[float] = None  # 0..1
    odometer_image: Optional[str] = None         # evidence image file name (saved in config.EVIDENCE)
    warning_lights: list[str] = []               # e.g. ["check engine", "airbag"]
    interior_damage: list[str] = []              # e.g. ["torn driver seat"]
    notes: list[str] = []                        # anything a reviewer should know


@dataclass(frozen=True)
class _OdometerCandidate:
    value: int
    confidence: float
    score: float
    box: tuple[int, int, int, int]


_reader = None
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _preprocess(image: np.ndarray) -> tuple[list[np.ndarray], float]:
    height, width = image.shape[:2]
    scale = min(2.0, 1200 / max(height, width)) if max(height, width) < 1200 else 1.0
    if scale > 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [enhanced, thresholded], scale


def _select_candidate(results, image_shape: tuple[int, int], scale: float = 1.0):
    """Choose a plausible odometer-like OCR result, mapping its box to source pixels."""
    height, width = image_shape
    candidates = []
    for result in results:
        if len(result) < 3:
            continue
        polygon, text, confidence = result[:3]
        digits = re.sub(r"\D", "", str(text))
        try:
            confidence = float(confidence)
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if not 4 <= len(digits) <= 7 or not 0 <= confidence <= 1 or confidence < 0.5:
            continue
        x1 = max(0, min(width, int(points[:, 0].min() / scale)))
        y1 = max(0, min(height, int(points[:, 1].min() / scale)))
        x2 = max(0, min(width, int(points[:, 0].max() / scale)))
        y2 = max(0, min(height, int(points[:, 1].max() / scale)))
        if x2 <= x1 or y2 <= y1:
            continue

        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        center_score = max(0.0, 1.0 - 2.0 * np.hypot(center_x - 0.5, center_y - 0.5))
        area_score = min(1.0, np.sqrt(((x2 - x1) * (y2 - y1)) / (width * height)) * 5)
        length_score = {4: 0.72, 5: 1.0, 6: 0.98, 7: 0.82}[len(digits)]
        score = 0.50 * confidence + 0.28 * length_score + 0.17 * center_score + 0.05 * area_score
        candidates.append(_OdometerCandidate(int(digits), confidence, score, (x1, y1, x2, y2)))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if len(candidates) > 1:
        runner_up = candidates[1]
        if runner_up.value != candidates[0].value and candidates[0].score - runner_up.score < 0.035:
            return None
    best = candidates[0]
    return best if best.score >= 0.55 else None


def _evidence_crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(12, (x2 - x1) // 2)
    pad_y = max(12, (y2 - y1) // 2)
    return image[max(0, y1 - pad_y):min(height, y2 + pad_y),
                 max(0, x1 - pad_x):min(width, x2 + pad_x)]


def analyze_interior(media: list[Path], claim_id: str) -> InteriorReport:
    """Read a likely odometer value from interior images and save its evidence crop."""
    images = [Path(path) for path in media if Path(path).suffix.lower() in _IMAGE_EXTENSIONS]
    videos = [Path(path) for path in media if Path(path).suffix.lower() in _VIDEO_EXTENSIONS]
    notes = []
    if videos:
        notes.append("Video input is not supported; provide dashboard photographs instead.")
    if not images:
        if not videos:
            notes.append("No supported dashboard image was provided.")
        return InteriorReport(notes=notes)

    unreadable = 0
    readable = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            unreadable += 1
            continue
        readable.append((image, _preprocess(image)))

    if not readable:
        notes.append("No readable dashboard images were supplied.")
        return InteriorReport(notes=notes)

    try:
        reader = _get_reader()
    except Exception as exc:
        notes.append(f"EasyOCR could not be initialized: {exc}")
        return InteriorReport(notes=notes)

    found = []
    for image, (variants, scale) in readable:
        for variant in variants:
            try:
                results = reader.readtext(variant, allowlist="0123456789", detail=1, paragraph=False)
            except Exception:
                continue
            candidate = _select_candidate(results, image.shape[:2], scale)
            if candidate:
                found.append((candidate, image))

    if not found:
        notes.append("No reliable 4-7 digit odometer reading was detected; the value was left blank.")
        if unreadable:
            notes.append(f"Skipped {unreadable} unreadable image(s).")
        return InteriorReport(notes=notes)

    found.sort(key=lambda item: item[0].score, reverse=True)
    best, image = found[0]
    conflicting = next((candidate for candidate, _ in found[1:]
                        if candidate.value != best.value and best.score - candidate.score < 0.035), None)
    if conflicting:
        notes.append("Dashboard images produced conflicting odometer readings; the value was left blank.")
        return InteriorReport(notes=notes)

    safe_claim_id = re.sub(r"[^A-Za-z0-9_-]", "_", claim_id).strip("_-") or "inspection"
    evidence_name = f"odometer_{safe_claim_id}_{uuid4().hex}.jpg"
    config.EVIDENCE.mkdir(parents=True, exist_ok=True)
    crop = _evidence_crop(image, best.box)
    if crop.size == 0 or not cv2.imwrite(str(config.EVIDENCE / evidence_name), crop):
        notes.append("The odometer was read, but its evidence image could not be saved.")
        evidence_name = None
    if unreadable:
        notes.append(f"Skipped {unreadable} unreadable image(s).")
    return InteriorReport(odometer_km=best.value, odometer_confidence=best.confidence,
                          odometer_image=evidence_name, notes=notes)
