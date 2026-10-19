import json
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import chassis, config, db
from .interior_api import router as interior_router
from .pipeline_exts import IMAGE_EXTS
from .schemas import ClaimOut, Stats, TrackEvidence
from .verification import inspect, inspect_images, verify


@asynccontextmanager
async def lifespan(app):
    db.init()
    yield


app = FastAPI(title="DiVision API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/evidence", StaticFiles(directory=config.EVIDENCE), name="evidence")
app.include_router(interior_router)


def _out(r) -> ClaimOut:
    earlier = [e for e in db.earlier_for_chassis(r["id"], r["chassis_no"]) if e["created_at"] <= r["created_at"]]
    damaged = next((e["id"] for e in earlier if e["verdict"] == "DAMAGED"), None)
    return ClaimOut(
        id=r["id"], vehicle=r["vehicle"], claim_text=r["claim_text"], status=r["status"],
        chassis_no=r["chassis_no"], prior_inspections=len(earlier), prior_damaged=damaged,
        verdict=r["verdict"], confidence=r["confidence"], action=r["action"], reason=r["reason"],
        detected_summary=r["detected_summary"], error=r["error"], demo=bool(r["demo"]),
        tracks=[TrackEvidence(**t) for t in json.loads(r["tracks_json"] or "[]")],
        created_at=r["created_at"])


def process_claim(claim_id: str):
    """Background job: run the CV pipeline on the stored video, then the claim-evidence engine."""
    row = db.get(claim_id)
    try:
        from . import pipeline  # lazy: torch/ultralytics are heavy
        target = config.UPLOADS / row["video_path"]
        if target.is_dir():  # photo set
            tracks, n = pipeline.run_images(target, claim_id)
            v = inspect_images(tracks, n)
        else:
            tracks, analyzed = pipeline.run(target, claim_id)
            # blank claim = pre-insurance inspection (any damage counts); otherwise claim-vs-evidence
            v = verify(row["claim_text"], tracks) if row["claim_text"].strip() else inspect(tracks, analyzed)
        # Fraud guard: a chassis that was flagged DAMAGED before must not be waved through as clean.
        prior = next((e["id"] for e in db.earlier_for_chassis(claim_id, row["chassis_no"])
                      if e["verdict"] == "DAMAGED"), None)
        if prior and v.verdict == "CLEAN":
            v = v.model_copy(update=dict(
                verdict="UNCERTAIN", action="Physical Inspection",
                reason=f"No damage found now, but this chassis was flagged DAMAGED in {prior}. "
                       "Possible hidden damage or a different vehicle; verify in person."))
        db.update(claim_id, status="done", verdict=v.verdict, confidence=v.confidence, action=v.action,
                  reason=v.reason, detected_summary=v.detected_summary, tracks=tracks)
    except Exception as e:  # surfaced in the dashboard instead of silently dropped
        db.update(claim_id, status="failed", error=str(e))


@app.post("/api/claims", response_model=ClaimOut)
async def create_claim(background: BackgroundTasks, chassis_no: str = Form(...), vehicle: str = Form(""),
                       claim_text: str = Form(""), files: list[UploadFile] = File(...)):
    try:
        chassis_no = chassis.normalize(chassis_no)
    except ValueError as e:
        raise HTTPException(422, str(e))
    claim_id = db.next_id()
    while (config.UPLOADS / claim_id).exists():  # never reuse an upload folder left over from a deleted row
        claim_id = f"CLM-{int(claim_id[4:]) + 1}"
    folder = config.UPLOADS / claim_id
    folder.mkdir()
    saved = []
    for up in files:
        name = Path(up.filename or "upload").name.replace(" ", "_")
        with open(folder / name, "wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append(name)
    images = [n for n in saved if Path(n).suffix.lower() in IMAGE_EXTS]
    if images and len(images) != len(saved):
        raise HTTPException(422, "Upload either one video or a set of photos, not both.")
    if not images and len(saved) != 1:
        raise HTTPException(422, "Upload a single video.")
    label = saved[0] + (f" (+{len(saved) - 1} more)" if len(saved) > 1 else "")
    target = claim_id if images else f"{claim_id}/{saved[0]}"
    db.insert(dict(id=claim_id, chassis_no=chassis_no, vehicle=vehicle or label, claim_text=claim_text, video_path=target,
                   status="processing", created_at=db.now(), demo=0))
    background.add_task(process_claim, claim_id)
    return _out(db.get(claim_id))


@app.get("/api/claims", response_model=list[ClaimOut])
def list_claims(verdict: Optional[str] = None):
    rows = [_out(r) for r in db.list_all()]
    return [r for r in rows if not verdict or r.verdict == verdict.upper()]


@app.get("/api/claims/{claim_id}", response_model=ClaimOut)
def get_claim(claim_id: str):
    r = db.get(claim_id)
    if not r:
        raise HTTPException(404, "Claim not found")
    return _out(r)


@app.get("/api/stats", response_model=Stats)
def stats():
    rows = db.list_all()
    count = lambda v: sum(1 for r in rows if r["verdict"] == v)
    return Stats(total=len(rows), match=count("MATCH"), mismatch=count("MISMATCH"),
                 clean=count("CLEAN"), damaged=count("DAMAGED"),
                 uncertain=count("UNCERTAIN"), processing=sum(1 for r in rows if r["status"] == "processing"))


@app.get("/api/health")
def health():
    return {"ok": True, "damage_weights": config.DAMAGE_WEIGHTS.exists(),
            "parts_weights": config.PARTS_WEIGHTS.exists()}
