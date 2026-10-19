"""HTTP endpoint for interior analysis. Owned by the interior teammate; main.py only includes the router."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from . import interior

router = APIRouter(prefix="/api/interior", tags=["interior"])


@router.post("/analyze", response_model=interior.InteriorReport)
def analyze(files: list[UploadFile] = File(...)):
    """Upload dashboard / interior photos (or one video) and get an InteriorReport back."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for up in files:
            p = Path(tmp) / Path(up.filename or "upload").name
            with open(p, "wb") as f:
                shutil.copyfileobj(up.file, f)
            paths.append(p)
        return interior.analyze_interior(paths, claim_id="adhoc")
