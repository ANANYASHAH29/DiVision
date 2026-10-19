# DiVision: pre-insurance vehicle inspection

Insurers price policies assuming most vehicles are in good condition. Customers who insure an already damaged
vehicle and claim a few days later cost the insurer money. DiVision inspects the vehicle **before** the policy is
issued, from a walk-around video or a set of photos, and decides: **Clean** (approve), **Damaged** (reject or
re-price) or **Needs review**.

```
video / photos -> YOLO11-Seg (detect + segment damage) -> BoT-SORT (track across frames)
               -> consistency checks -> inspection verdict -> insurer dashboard
```

**The exterior model is already trained and included in this repo. You do not need to train anything.**

---

# START HERE (teammate)

## Part 1: set up (about 20 minutes)
You need **Git**, **Python 3.10-3.12** and **Node.js 20+**. Check: `git --version`, `python --version`, `node --version`.
Commands are for Windows PowerShell. On Mac/Linux use `python3` and `source .venv/bin/activate`.

**Step 1. Get the code**
```
git clone https://github.com/ANANYASHAH29/DiVision.git
cd DiVision
git checkout -b feature/interior
```

**Step 2. Python environment** (the install is large because of PyTorch; this is normal and one-time)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

**Step 3. Run the tests**
```
cd backend
python -m pytest -q tests
cd ..
```
Expected: `12 passed`.

**Step 4. Start the backend** (terminal 1, from the repo root; leave it running)
```
.venv\Scripts\activate
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/api/health . Expected: `{"ok":true,"damage_weights":true,"parts_weights":false}`.
(`parts_weights: false` is normal.)

**Step 5. Start the dashboard** (terminal 2)
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 . You should see the DiVision dashboard.

## Part 2: check everything works (5 minutes)
Tick each box before you start your task. If one fails, fix it or message Ananya.

- [ ] `python -m pytest -q backend/tests` prints `12 passed`
- [ ] http://127.0.0.1:8000/api/health shows `"damage_weights":true`
- [ ] http://localhost:5173 shows the dashboard
- [ ] **Try the product:** in the dashboard type chassis number `MA3EWDE1S00123456`, click "Choose video or photos"
      and upload a video of a car (or 4+ photos of a car). The row shows ANALYSING, then a result (CLEAN, DAMAGED or
      NEEDS REVIEW). Click the row to see the evidence. On a laptop without a GPU a short video takes a few minutes.
- [ ] **Try your module's starting point:** `python scripts/run_interior.py any_photo.jpg` prints an empty report with
      the note "interior analysis not implemented yet". That is your starting point.

## Part 3: your assignment (interior + odometer)
**Goal:** read the vehicle's odometer (km) from a dashboard photo, so we can catch odometer rollback fraud.
**You do not train models.** Use pretrained OCR (EasyOCR first) and, if needed, the vision-LLM fallback described in
the guide. If accuracy is poor, improve preprocessing or try another OCR engine, and tell Ananya.

Do these in order. The full detail is in [docs/TEAMMATE_INTERIOR.md](docs/TEAMMATE_INTERIOR.md).

1. **Collect test data.** 20-30 dashboard/odometer photos (your own car, family cars, free photos from Pexels or
   Unsplash). Write the true km of each in a local file `my_odometer_tests.csv` (do not commit it).
2. **Implement `analyze_interior()`** in `backend/app/interior.py` with EasyOCR. Keep the function name and the
   `InteriorReport` fields. Add `easyocr` to `backend/requirements.txt`.
3. **Measure accuracy** on your test photos (exact matches out of total) and write the number down. It goes on a slide.
4. **Rollback check.** If a new inspection's reading is lower than an earlier inspection of the same chassis, flag it.
5. **Dashboard panel.** Create `frontend/src/InteriorPanel.jsx` showing the reading and evidence crop; import it in
   `App.jsx` with one line.
6. **Report back.** `git add` your files, `git commit`, `git push -u origin feature/interior`, open a Pull Request on
   GitHub, and message Ananya with your accuracy number.

**Rules:** add new files; do not edit `pipeline.py`, `verification.py`, `reflection.py`, `verifier.py` or the
training scripts. Never commit real customer photos, plates, databases, API keys or model weights other than the one
already included. Check `git status` before every commit. Stuck for 30 minutes? Ask.

---

# Reference

## What is in the box
* **Damage detection:** YOLO11-Seg fine-tuned on CarDD (6 classes: dent, scratch, crack, glass shatter, lamp broken,
  tire flat). Mask mAP@50 0.747 on the CarDD test split.
* **Tracking:** BoT-SORT keeps one identity per damage across frames, so one dent is not counted 40 times.
* **Chassis number (VIN)** is required for every inspection, with repeat-submission detection: a chassis flagged
  DAMAGED earlier can never come back CLEAN without a physical check.
* **Coverage rules:** a vehicle is only certified clean if enough of it was filmed (enough frames, or >= 4 photos).
* **Dashboard:** React + Tailwind insurer queue with evidence frames per damage.

## Known limitations (honest)
* Reflections and water droplets on glass are sometimes reported as damage (see [docs/EVALUATION.md](docs/EVALUATION.md)).
  Retraining with better data is being done by Ananya.
* No parts model yet, so damage is not tied to a named part.
* CPU inference is slow (about 0.5 s per frame). A GPU or an OpenVINO export is needed for real-time use.

## Layout
`backend/app` API, pipeline and rules - `frontend/src` dashboard - `scripts` data prep and evaluation -
`docs` team notes - `data/weights/damage_seg.pt` the trained model.

## Maintainers only: retraining (teammates can skip this section)
1. Download CarDD (5.6 GB, https://cardd-ustc.github.io; the authors' Google Drive file is
   `gdown 1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx -O CarDD_release.zip`) and extract to `data/raw/CarDD_release`.
2. `python scripts/prepare_cardd.py --src data/raw/CarDD_release`
3. Train on a GPU (Colab/Kaggle): `python scripts/train.py --epochs 100 --device 0`
