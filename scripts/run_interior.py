"""Run interior analysis (odometer etc.) on local files, without the API or the exterior model.

Usage:  python scripts/run_interior.py dashboard1.jpg [dashboard2.jpg ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.interior import analyze_interior  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
print(analyze_interior([Path(a) for a in sys.argv[1:]], "cli").model_dump_json(indent=2))
