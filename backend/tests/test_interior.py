"""Contract tests for the interior module. Keep these passing while you implement it."""
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import interior
from app.interior import InteriorReport, analyze_interior


def _box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_returns_report_even_for_missing_input():
    r = analyze_interior([Path("does_not_exist.jpg")], "t")
    assert isinstance(r, InteriorReport)
    assert r.odometer_km is None
    assert any("No readable" in note for note in r.notes)


def test_candidate_selection_prefers_centered_odometer_and_ignores_short_numbers():
    candidates = [
        (_box(20, 20, 150, 55), "120", 0.99),
        (_box(30, 120, 180, 155), "98765", 0.72),
        (_box(130, 85, 290, 120), "54321", 0.91),
    ]

    chosen = interior._select_candidate(candidates, (200, 320))

    assert chosen.value == 54321
    assert chosen.confidence == 0.91
    assert chosen.box == (130, 85, 290, 120)


def test_candidate_selection_rejects_ambiguous_readings():
    candidates = [
        (_box(100, 80, 220, 115), "54321", 0.85),
        (_box(100, 80, 220, 115), "54327", 0.84),
    ]

    assert interior._select_candidate(candidates, (200, 320)) is None


def test_preprocess_preserves_source_scale_for_large_images():
    image = np.zeros((1300, 1800, 3), dtype=np.uint8)

    variants, scale = interior._preprocess(image)

    assert scale == 1.0
    assert all(variant.shape == image.shape[:2] for variant in variants)


def test_analyze_saves_unique_claim_scoped_evidence(monkeypatch, tmp_path):
    image_path = tmp_path / "dashboard.jpg"
    cv2.imwrite(str(image_path), np.full((200, 320, 3), 127, dtype=np.uint8))
    evidence_dir = tmp_path / "evidence"
    monkeypatch.setattr(interior.config, "EVIDENCE", evidence_dir)

    class FakeReader:
        def readtext(self, image, **kwargs):
            assert kwargs["allowlist"] == "0123456789"
            return [(_box(130, 85, 290, 120), "54321", 0.91)]

    monkeypatch.setattr(interior, "_get_reader", lambda: FakeReader())

    first = analyze_interior([image_path], "cli")
    second = analyze_interior([image_path], "cli")

    assert first.odometer_km == 54321
    assert first.odometer_confidence == 0.91
    assert first.odometer_image.startswith("odometer_cli_")
    assert Path(first.odometer_image).name == first.odometer_image
    assert first.odometer_image != second.odometer_image
    assert (evidence_dir / first.odometer_image).is_file()
    assert (evidence_dir / second.odometer_image).is_file()


def test_video_input_returns_clear_note_without_ocr(monkeypatch, tmp_path):
    def unexpected_reader():
        raise AssertionError("reader should not be initialized for video-only input")

    monkeypatch.setattr(interior, "_get_reader", unexpected_reader)

    report = analyze_interior([tmp_path / "walkaround.mp4"], "claim")

    assert report.odometer_km is None
    assert any("Video input is not supported" in note for note in report.notes)


def test_api_endpoint_roundtrip():
    from app.main import app
    with TestClient(app) as c:
        res = c.post("/api/interior/analyze", files=[("files", ("d.jpg", b"not a real image", "image/jpeg"))])
    assert res.status_code == 200
    body = res.json()
    assert {"odometer_km", "odometer_confidence", "warning_lights", "interior_damage", "notes"} <= set(body)
