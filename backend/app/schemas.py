from typing import Optional
from pydantic import BaseModel


class TrackEvidence(BaseModel):
    track_id: int
    damage: str
    component: Optional[str] = None
    max_conf: float
    mean_conf: float
    n_frames: int
    first_frame: int
    best_frame: int
    mask_area_px: float = 0.0
    evidence_image: Optional[str] = None
    image_name: Optional[str] = None   # set when the detection came from a still photo
    reflection_risk: float = 0.0       # 0..1, see reflection.py
    likely_reflection: bool = False    # excluded from the damage count and from a DAMAGED verdict
    flags: list[str] = []
    real_prob: Optional[float] = None  # verifier score: probability this is real damage


class Verdict(BaseModel):
    verdict: str            # MATCH | MISMATCH | UNCERTAIN | CLEAN | DAMAGED
    confidence: Optional[float] = None
    action: str             # Remote Verification | Manual Review | Physical Inspection
    reason: str
    claimed_component: Optional[str] = None
    claimed_damage: Optional[str] = None
    detected_summary: str


class ClaimOut(BaseModel):
    id: str
    vehicle: str
    chassis_no: Optional[str] = None
    prior_inspections: int = 0          # earlier completed inspections of the same chassis
    prior_damaged: Optional[str] = None  # id of an earlier inspection of this chassis that was DAMAGED
    claim_text: str
    status: str             # processing | done | failed
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    action: Optional[str] = None
    reason: Optional[str] = None
    detected_summary: Optional[str] = None
    tracks: list[TrackEvidence] = []
    error: Optional[str] = None
    demo: bool = False
    created_at: str


class Stats(BaseModel):
    total: int
    match: int
    mismatch: int
    uncertain: int
    clean: int = 0
    damaged: int = 0
    processing: int
