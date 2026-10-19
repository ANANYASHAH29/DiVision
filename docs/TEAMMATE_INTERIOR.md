# Teammate guide: interior + odometer module (step by step)

DiVision is a pre-insurance inspection platform: before a policy is issued, the insurer wants to know if the
vehicle is already damaged. The **exterior** part (damage detection, tracking, fraud checks) is done and is being
retrained by Ananya. **Your job is the interior**: read the odometer, and optionally warning lights and interior
condition.

**Why the odometer matters:** people roll back odometers to get cheaper cover. We read the km from the dashboard and
compare it with the previous inspection of the *same chassis number*. A lower reading than before is a fraud flag.

**You do NOT need to train a model to start.** Use a pretrained OCR library first. Only train a small detector if
the baseline is not accurate enough (see step 6).

---

## Part 1: set up your computer (about 20 minutes, one time)

You need: **Git**, **Python 3.10, 3.11 or 3.12**, **Node.js 20+**. Check with:
```
git --version
python --version
node --version
```

**1. Get the code** (ask Ananya to add you as a collaborator on GitHub first)
```
git clone https://github.com/ANANYASHAH29/DiVision.git
cd DiVision
git checkout -b feature/interior
```

**2. Python environment** (Windows PowerShell; on Mac/Linux use `source .venv/bin/activate` and `python3`)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```
The first install is large (PyTorch, about 1-2 GB). This is normal. You need it because the backend imports the
exterior pipeline, but you will not run that model.

**3. Check the backend works**
```
cd backend
python -m pytest -q tests
```
Expect `12 passed`. If you see that, your setup is correct.

**4. Start the backend and the dashboard** (two terminals, both from the repo root)
```
# terminal 1
.venv\Scripts\activate
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# terminal 2
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 . You should see the DiVision dashboard. Open http://127.0.0.1:8000/docs to see the API;
find **POST /api/interior/analyze**, that is your endpoint.

The trained exterior model (`data/weights/damage_seg.pt`) is already in the repo, so you can also try a full
exterior inspection: upload a car video or photos with any 17-character chassis number such as
`MA3EWDE1S00123456`. You do not need it for the interior work.

---

## Part 2: what to build

Everything you write goes in these places:

| File | What it is |
|---|---|
| `backend/app/interior.py` | **Your main file.** Implement `analyze_interior(media, claim_id) -> InteriorReport`. Keep the function name and the `InteriorReport` fields. |
| `backend/app/interior_api.py` | The HTTP endpoint. Already works; it calls your function. |
| `scripts/run_interior.py` | Run your code on local photos from the command line. |
| `backend/tests/test_interior.py` | Contract tests. Keep them passing; add your own in new files. |
| `frontend/src/` | Add a new component file (e.g. `InteriorPanel.jsx`) for the dashboard. |

Try the stub right now:
```
python scripts/run_interior.py some_photo.jpg
```
It prints an empty report. Your goal is to make it print the odometer reading.

### Step 3: collect test data (do this first)
Get **20-30 odometer photos**: your own car, family cars, a few from Pexels/Unsplash (search "car dashboard
odometer"). Include a mix: digital and analog displays, day and night, straight and slightly angled.
Write the true reading of each photo in a file `my_odometer_tests.csv` (keep it local, do not commit it). This is
how you measure whether your reader works, so do not skip it.

### Step 4: baseline with pretrained OCR
1. `pip install easyocr` and add `easyocr` to `backend/requirements.txt`.
2. In `analyze_interior`, load each image, run EasyOCR with a digits-only allowlist
   (`reader.readtext(img, allowlist="0123456789")`), and pick the most plausible number: 1 to 7 digits,
   largest confident box in the centre of the dashboard cluster. Ignore the trip meter (usually smaller and
   labelled A/B).
3. Return `odometer_km`, `odometer_confidence` and save a cropped evidence image into `config.EVIDENCE`.
4. Measure: what percentage of your test photos does it read exactly right? Write this number down. It goes on a
   presentation slide.

### Step 5: improve the baseline (pick what helps)
Image preprocessing (grayscale, contrast, upscale, deskew), try Tesseract or PaddleOCR, handle both km and miles,
and return `None` with a note when the reading is not trustworthy. **Never guess**: a wrong reading is worse than
no reading.

### Step 6: only if accuracy is poor: train a small detector
If the baseline is below about 80%, train a small YOLO to find the odometer display first, then OCR the crop.
Search Roboflow Universe for an "odometer" or "dashboard" dataset, train a nano model on Google Colab
(`pip install ultralytics`, `YOLO("yolo11n.pt").train(...)`; ask Ananya how she trained on Colab).

### Step 7: rollback check
Add an `odometer_km` column to the database the same way `chassis_no` was added in `db.init()` in
`backend/app/db.py`. When a new inspection has a reading **lower** than an earlier inspection of the same chassis
(`db.earlier_for_chassis(...)`), add a note and flag it. Talk to Ananya before editing `db.py`/`main.py`.

### Step 8: dashboard panel
Create `frontend/src/InteriorPanel.jsx` showing the reading, the evidence crop, warning lights and notes, and
import it in `App.jsx` with one line. Keep your changes small so merges stay easy.

### Optional: a faster option for the demo
A vision-capable LLM API (e.g. Claude) can read an odometer from a photo in a few lines of code. It is a good
fallback and quick to demo, but it sends images to an external service and needs an API key. Never commit keys
(`.env` is git-ignored) and never send real customer photos. Use it only with your own test photos.

---

## Part 3: working rules (so we do not break each other's work)
* Work on **your branch** `feature/interior`; open a Pull Request into `main` when something works.
* **Add new files; do not edit** `pipeline.py`, `verification.py`, `reflection.py`, `verifier.py` or anything in
  `scripts/` except `run_interior.py`. They belong to the exterior model work.
* **Never commit** real customer photos, vehicle photos with plates, databases, model weights or API keys.
  `.gitignore` blocks the common ones, but check `git status` before every commit.
* Commit small and often: `git add <files>`, `git commit -m "interior: read odometer with EasyOCR"`,
  `git push -u origin feature/interior`.
* Stuck for more than 30 minutes? Ask. Ananya can answer pipeline questions quickly.

## Definition of done
1. `python scripts/run_interior.py photo.jpg` prints the correct km for most of your test photos.
2. Your measured accuracy number is written down (say 18 of 24 photos read exactly).
3. `python -m pytest -q backend/tests` passes.
4. The dashboard shows the reading for an inspection, and a lower-than-before reading is flagged.
