# Teammate guide: interior + odometer module

DiVision is a pre-insurance inspection platform. Before a policy is issued, the insurer wants to know whether
the vehicle is already damaged. The exterior part (damage detection, tracking, fraud checks) is done. Your job is
the **interior**: odometer reading, warning lights and interior condition.

## Why the odometer matters
People roll back odometers to get cheaper cover. We read the km from the dashboard and compare it with the
previous inspection of the **same chassis number**. A reading lower than before is a fraud flag.

## What you build
Implement `analyze_interior()` in [backend/app/interior.py](../backend/app/interior.py). Keep the function
signature and the `InteriorReport` fields. Everything else is yours:

1. **Odometer OCR** (EasyOCR or Tesseract, add it to `backend/requirements.txt`). Locate the display, read the
   digits, return `odometer_km` and a confidence. Save a cropped evidence image into `config.EVIDENCE`.
2. **Warning lights** (check engine, airbag, ABS...). Optional.
3. **Interior condition** (seats, dashboard). Optional.
4. **Rollback rule**: in your own function, compare against earlier inspections of the same chassis
   (`db.earlier_for_chassis(...)`; add an `odometer_km` column the way `chassis_no` was added in `db.init()`).
5. **Dashboard panel**: a new component in `frontend/src/` shown in the inspection detail drawer. Add it as a new
   file and a one-line import, so we do not conflict in `App.jsx`.

## Rules to avoid clashing with the rest of the team
* Work on a branch: `git checkout -b feature/interior`, open a pull request into `main`.
* **Add new files; do not edit** `pipeline.py`, `verification.py`, `reflection.py` or anything in `scripts/`.
  Those belong to the exterior model work, which is being retrained.
* Never commit photos of real customers or vehicles, databases or model weights. `.gitignore` blocks them.
  Test with your own dashboard photos.

## Run it locally
```
python -m venv .venv
.venv\Scripts\pip install -r backend/requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
cd frontend && npm install && npm run dev        # http://localhost:5173
cd backend && ..\.venv\Scripts\python -m pytest tests
```
The exterior model needs `data/weights/damage_seg.pt` (20 MB). Ask Ananya for it and drop it in that folder;
you do not need it to work on the interior module.
