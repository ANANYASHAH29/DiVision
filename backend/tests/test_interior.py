"""Contract tests for the interior module. Keep these passing while you implement it."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.interior import InteriorReport, analyze_interior


def test_returns_report_even_for_missing_input():
    r = analyze_interior([Path("does_not_exist.jpg")], "t")
    assert isinstance(r, InteriorReport)


def test_api_endpoint_roundtrip():
    from app.main import app
    with TestClient(app) as c:
        res = c.post("/api/interior/analyze", files=[("files", ("d.jpg", b"not a real image", "image/jpeg"))])
    assert res.status_code == 200
    body = res.json()
    assert {"odometer_km", "odometer_confidence", "warning_lights", "interior_damage", "notes"} <= set(body)
