"""Decide whether a tracked detection is probably a reflection / glare / false alarm instead of damage.

Two independent kinds of evidence are combined:

* verifier  - second-stage classifier on the detection's crop (see verifier.py). A real_prob below its
              threshold is strong evidence on its own.
* temporal  - real damage is attached to the car: it keeps the same label and stays continuously in view as
              the camera moves. A reflection flips label or flickers in and out as the viewing angle changes.
              One temporal sign alone is not enough, two together are.

Tracks flagged here are not counted as damage and cannot trigger a DAMAGED verdict by themselves; they
send the inspection to review instead (they are never silently turned into CLEAN).
"""
from collections import Counter
from typing import Optional

VERIFIER_WEIGHT = 0.6
LABEL_WEIGHT = 0.3
FLICKER_WEIGHT = 0.3
MIN_PURITY, MIN_PERSISTENCE = 0.7, 0.6
LIKELY_AT = 0.5


def assess(classes: list[str], idx: Optional[list[int]] = None, real_prob: Optional[float] = None,
           verifier_threshold: Optional[float] = None) -> dict:
    """`idx` = analysed-frame index of each sighting (video only). Returns risk, flags and a verdict."""
    flags: list[str] = []
    risk = 0.0

    if real_prob is not None and verifier_threshold is not None and real_prob < verifier_threshold:
        flags.append(f"verifier says this does not look like real damage (score {real_prob:.2f})")
        risk += VERIFIER_WEIGHT

    if len(classes) >= 3 and Counter(classes).most_common(1)[0][1] / len(classes) < MIN_PURITY:
        flags.append("label keeps changing between damage types")
        risk += LABEL_WEIGHT

    if idx and len(idx) >= 3 and len(idx) / (idx[-1] - idx[0] + 1) < MIN_PERSISTENCE:
        flags.append("flickers in and out of view")
        risk += FLICKER_WEIGHT

    risk = min(risk, 1.0)
    return dict(risk=round(risk, 2), flags=flags, likely=risk >= LIKELY_AT)
