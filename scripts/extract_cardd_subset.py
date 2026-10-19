"""Extract only the CarDD COCO annotations plus the val/test images (~0.9 GB) from the release zip.

Usage:  python scripts/extract_cardd_subset.py C:\\Users\\you\\Downloads\\CarDD_release.zip
"""
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/cardd_subset")
want = ("CarDD_COCO/annotations/", "CarDD_COCO/val2017/", "CarDD_COCO/test2017/")

with zipfile.ZipFile(zip_path) as zf:
    members = [i for i in zf.infolist() if not i.is_dir() and any(w in i.filename for w in want)]
    for n, info in enumerate(members, 1):
        rel = info.filename.split("CarDD_COCO/", 1)[1]
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(zf.read(info))
        if n % 200 == 0:
            print(n, "/", len(members))
print("done ->", out.resolve())
