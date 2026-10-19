"""CLI: run the full pipeline on one video without the API.

Usage:  python scripts/run_video.py walkaround.mp4 ["Rear bumper scratch"]   (no claim = pre-insurance inspection)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import pipeline  # noqa: E402
from app.verification import inspect, verify  # noqa: E402

video, claim = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "")
tracks, analyzed = pipeline.run(video, video.stem)
for t in tracks:
    print(t.model_dump())
print((verify(claim, tracks) if claim else inspect(tracks, analyzed)).model_dump_json(indent=2))
