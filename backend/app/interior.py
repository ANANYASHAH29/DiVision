"""Interior inspection: odometer reading, warning lights, interior condition.   ** OWNER: teammate **

This module is a stub so the rest of the system can already depend on its interface. Implement
`analyze_interior` and keep the signature and the `InteriorReport` fields; see docs/TEAMMATE_INTERIOR.md.
"""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class InteriorReport(BaseModel):
    odometer_km: Optional[int] = None            # None = could not be read
    odometer_confidence: Optional[float] = None  # 0..1
    odometer_image: Optional[str] = None         # evidence image file name (saved in config.EVIDENCE)
    warning_lights: list[str] = []               # e.g. ["check engine", "airbag"]
    interior_damage: list[str] = []              # e.g. ["torn driver seat"]
    notes: list[str] = []                        # anything a reviewer should know


def analyze_interior(media: list[Path], claim_id: str) -> InteriorReport:
    """Analyse dashboard / interior photos or a video and return what was found.

    `media` is the list of uploaded interior files (images or one video). Not implemented yet: return an
    empty report so callers can run end to end.
    """
    return InteriorReport(notes=["interior analysis not implemented yet"])
