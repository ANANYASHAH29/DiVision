import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
EVIDENCE = DATA / "evidence"
DB_PATH = Path(os.getenv("DIVISION_DB", DATA / "division.db"))

# Damage model trained on CarDD (6 classes). Parts model is optional.
DAMAGE_WEIGHTS = Path(os.getenv("DIVISION_DAMAGE_WEIGHTS", DATA / "weights" / "damage_seg.pt"))
PARTS_WEIGHTS = Path(os.getenv("DIVISION_PARTS_WEIGHTS", DATA / "weights" / "parts_seg.pt"))

DAMAGE_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]

CONF_THRESHOLD = 0.25
MIN_TRACK_FRAMES = 3      # a damage track must persist this many frames to count as evidence
STRIDE = 2                # process every Nth frame (TVDT has ~1,700 frames per video)
MIN_PHOTOS = 4            # photos needed before a 'clean' verdict is trusted (one per side)
MIN_COVERAGE_FRAMES = 60  # analysed frames needed before a 'clean' verdict is trusted
STRONG_CONF = 0.5         # track max-confidence needed for a confident verdict

for d in (UPLOADS, EVIDENCE, DATA / "weights"):
    d.mkdir(parents=True, exist_ok=True)
