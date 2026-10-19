import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY, vehicle TEXT, claim_text TEXT, video_path TEXT,
  status TEXT, verdict TEXT, confidence REAL, action TEXT, reason TEXT,
  detected_summary TEXT, tracks_json TEXT DEFAULT '[]', error TEXT,
  demo INTEGER DEFAULT 0, created_at TEXT
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(claims)")}
        if "chassis_no" not in cols:  # migrate databases created before chassis numbers existed
            c.execute("ALTER TABLE claims ADD COLUMN chassis_no TEXT")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def next_id() -> str:
    # max+1 rather than COUNT so deleting a row can never cause an id (and upload folder) collision
    with conn() as c:
        n = c.execute("SELECT MAX(CAST(SUBSTR(id, 5) AS INTEGER)) FROM claims").fetchone()[0]
    return f"CLM-{(n or 2299) + 1}"


def earlier_for_chassis(claim_id: str, chassis_no: str):
    """Other completed inspections of the same chassis, oldest first."""
    if not chassis_no:
        return []
    with conn() as c:
        return c.execute(
            "SELECT id, verdict, created_at FROM claims WHERE chassis_no=? AND id<>? AND status='done' "
            "ORDER BY created_at, id", (chassis_no, claim_id)).fetchall()


def insert(row: dict):
    with conn() as c:
        c.execute(f"INSERT INTO claims ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                  list(row.values()))


def update(claim_id: str, **fields):
    if "tracks" in fields:
        fields["tracks_json"] = json.dumps(
            [t if isinstance(t, dict) else t.model_dump() for t in fields.pop("tracks")])
    with conn() as c:
        c.execute(f"UPDATE claims SET {','.join(k + '=?' for k in fields)} WHERE id=?",
                  [*fields.values(), claim_id])


def get(claim_id: str):
    with conn() as c:
        return c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()


def list_all():
    with conn() as c:
        return c.execute("SELECT * FROM claims ORDER BY created_at DESC, id DESC").fetchall()
