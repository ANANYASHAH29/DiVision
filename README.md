# DiVision: pre-insurance vehicle inspection

Insurers price policies assuming most vehicles are in good condition. Customers who insure an already damaged
vehicle and claim a few days later cost the insurer money. DiVision inspects the vehicle **before** the policy is
issued from a walk-around video or a set of photos, and decides: **Clean** (approve), **Damaged** (reject or
re-price) or **Needs review**.

```
video / photos -> YOLO11-Seg (detect + segment damage) -> BoT-SORT (track across frames)
               -> reflection / consistency checks -> inspection verdict -> insurer dashboard
```

## What is in the box
* **Damage detection** - YOLO11-Seg fine-tuned on CarDD (6 classes: dent, scratch, crack, glass shatter, lamp broken,
  tire flat). Mask mAP@50 0.747 on the CarDD test split.
* **Tracking** - BoT-SORT keeps one identity per damage across frames, so one dent is not counted 40 times.
* **Chassis number (VIN)** required for every inspection, with repeat-submission detection: a chassis flagged DAMAGED
  earlier can never come back CLEAN without a physical check.
* **Coverage rules** - a vehicle is only certified clean if enough of it was filmed (enough frames or >= 4 photos).
* **Dashboard** - React + Tailwind insurer queue with evidence frames per damage.

## Known limitations (honest)
* Reflections and water droplets on glass are sometimes reported as damage. On our own test photos the raw model
  wrongly rejects about 22% of clean-car photos and 48% of reflection-heavy photos. Retraining with hard negatives is
  in progress. A second-stage verifier was tried and rejected because it also suppressed real damage.
* No parts model yet, so damage is not tied to a named part.
* CPU inference is slow (about 0.5 s per frame). A GPU or OpenVINO export is needed for real-time use.

## Setup, step by step
Requirements: Git, Python 3.10-3.12, Node.js 20+.

1. **Clone:** `git clone https://github.com/ANANYASHAH29/DiVision.git` then `cd DiVision`
2. **Python environment** (Windows PowerShell; Mac/Linux: `source .venv/bin/activate`):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r backend/requirements.txt      # large (PyTorch), one-time
   ```
3. **Check it works:** `cd backend` then `python -m pytest -q tests` should print `12 passed`.
4. **Start the backend** from the repo root (keep this terminal open):
   `python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000`
   API docs: http://127.0.0.1:8000/docs
5. **Start the dashboard** in a second terminal: `cd frontend`, `npm install`, `npm run dev`, then open
   http://localhost:5173
6. **Model weights:** the trained damage model `data/weights/damage_seg.pt` (20 MB) is included in the repo.
   Check http://127.0.0.1:8000/api/health shows `"damage_weights": true`. Then upload a video or photos in the
   dashboard (a 17-character chassis number is required, e.g. `MA3EWDE1S00123456`).
7. **Run on one video without the UI:** `python scripts/run_video.py video.mp4`

Interior / odometer work does not need the weights: see [docs/TEAMMATE_INTERIOR.md](docs/TEAMMATE_INTERIOR.md).

## Train
1. Download CarDD (5.6 GB, https://cardd-ustc.github.io; the authors' Google Drive file is
   `gdown 1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx -O CarDD_release.zip`) and extract to `data/raw/CarDD_release`.
2. `python scripts/prepare_cardd.py --src data/raw/CarDD_release`
3. Train on a GPU (Colab/Kaggle): `python scripts/train.py --epochs 100 --device 0`

## Layout
`backend/app` API, pipeline, rules - `frontend/src` dashboard - `scripts` data prep and evaluation -
`docs` team notes ([teammate guide](docs/TEAMMATE_INTERIOR.md), [evaluation results](docs/EVALUATION.md)).
