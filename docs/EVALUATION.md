# Evaluation: what we measured

All numbers below come from the scripts in this repo. Nothing here is estimated.

## 1. Detector accuracy (CarDD test split, 374 images, never used for training)
YOLO11s-seg fine-tuned for 100 epochs on CarDD train (2,816 images).

| Metric | Value |
|---|---|
| Mask mAP@50 | 0.747 |
| Mask mAP@50-95 | 0.570 |
| Box mAP@50 | 0.753 |
| Mask precision / recall | 0.83 / 0.69 |

Strong classes: glass shatter, lamp broken, tire flat. Weak classes: dent, scratch, crack (thin, low contrast).

## 2. False alarms on undamaged cars (the open problem)
We ran the detector on photos where every car is undamaged (49 "clean" street photos, 29 photos with strong
reflections or water droplets on glass). An image counts as wrongly rejected if any detection has
confidence >= 0.5, which is what triggers a DAMAGED verdict.

| Photo set | Images | Wrongly rejected as DAMAGED |
|---|---|---|
| Clean cars | 49 | 11 (22%) |
| Reflection / droplet heavy | 29 | 14 (48%) |

Typical mistakes: a reflective windshield outlined as "glass shatter" at 0.90+ confidence, water droplets on
a mirror, a headlight called "lamp broken", a building window called "crack".
Reproduce: see `scripts/verifier_mine.py` (stage 1 prints the detections).

## 3. Real broken windshields (263 photos of 131 vehicles, company inspection photos, not in this repo)
The model, never trained on trucks, flags at least one damage with confidence >= 0.5 in 151 of 263 images (57%);
glass shatter specifically is found in about 48%. It misses small cracks and cab-interior shots.

## 4. A fix that did NOT work: second-stage "reflection verifier"
We trained a small classifier on detector false alarms vs real damage (scripts/verifier_mine.py,
scripts/verifier_train.py). On held-out photos it cut false rejections (22% -> 8%, 48% -> 24%), but on a real
wrecked-car video it labelled a genuinely shattered windshield and a genuinely flat tyre as reflections.
Cause: all negative examples came from professional photography, so it partly learned "professional photo =
clean". In-domain it was also worse than the detector confidence alone (AUC 0.756 vs 0.816). **It is disabled**
(the code stays in `backend/app/verifier.py` and activates only if `data/weights/verifier.npz` exists).

Lesson: negatives and positives must come from the same capture style. The real fix is retraining the detector
with hard negatives (intact glass, reflections) from the **same source** as the damage photos, plus a held-out
test split by vehicle that includes a real wrecked-car video as a must-pass check.

## 5. Speed
On a laptop CPU (13th-gen i7-U, no GPU): decode 23 ms/frame, detector 475 ms/frame, detector + tracker
545 ms/frame. A 1-minute video at 1,500 frames is far too slow without a GPU or an OpenVINO/ONNX export.
