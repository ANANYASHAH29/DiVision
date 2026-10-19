"""Claim-Evidence engine: compares the claimed component/damage with tracked detections."""
import re
from typing import Optional

from .config import MIN_COVERAGE_FRAMES, MIN_PHOTOS, MIN_TRACK_FRAMES, STRONG_CONF
from .schemas import TrackEvidence, Verdict

COMPONENTS = {
    "front bumper": ["front bumper"],
    "rear bumper": ["rear bumper", "back bumper"],
    "bumper": ["bumper"],
    "bonnet": ["bonnet", "hood"],
    "boot": ["boot", "trunk", "tailgate"],
    "door": ["door", "side door"],
    "fender": ["fender", "wing", "quarter panel"],
    "headlight": ["headlight", "head light", "headlamp", "head lamp"],
    "tail light": ["tail-light", "tail light", "taillight", "tail lamp", "brake light"],
    "windshield": ["windshield", "windscreen", "front glass"],
    "rear glass": ["rear glass", "rear windshield", "back glass"],
    "window": ["window"],
    "mirror": ["mirror"],
    "wheel": ["wheel", "tyre", "tire", "rim"],
    "roof": ["roof"],
}
DAMAGES = {
    "dent": ["dent", "dented", "ding", "crumple"],
    "scratch": ["scratch", "scratched", "scrape", "scuff"],
    "crack": ["crack", "cracked", "fracture"],
    "glass shatter": ["shatter", "shattered", "smashed glass", "broken glass"],
    "lamp broken": ["lamp broken", "broken lamp", "broken light", "light broken", "broken headlight"],
    "tire flat": ["flat", "puncture", "deflated"],
}
# longest synonym first so "rear bumper" wins over "bumper"
_COMP_SYNS = sorted(((s, c) for c, ss in COMPONENTS.items() for s in ss), key=lambda x: -len(x[0]))
_DMG_SYNS = sorted(((s, d) for d, ss in DAMAGES.items() for s in ss), key=lambda x: -len(x[0]))

LAMP_COMPONENTS = {"headlight", "tail light"}
GLASS_COMPONENTS = {"windshield", "rear glass", "window"}


def _find(text: str, table) -> Optional[str]:
    t = text.lower().replace("_", " ")
    for syn, canon in table:
        if re.search(r"\b" + re.escape(syn) + r"\b", t):
            return canon
    return None


def parse_claim(text: str) -> tuple[Optional[str], Optional[str]]:
    return _find(text, _COMP_SYNS), _find(text, _DMG_SYNS)


def _norm_component(c: Optional[str]) -> Optional[str]:
    if not c:
        return None
    c = c.lower().replace("_", " ").strip()
    return _find(c, _COMP_SYNS) or c


def component_match(claimed: Optional[str], detected: Optional[str]) -> bool:
    if not claimed or not detected:
        return False
    if claimed == detected:
        return True
    # generic "bumper" matches front/rear bumper either way
    return "bumper" in claimed and "bumper" in detected and "bumper" in (claimed, detected)


def damage_match(claimed: Optional[str], track: TrackEvidence) -> bool:
    if not claimed:
        return False
    d = track.damage
    if claimed == d:
        return True
    comp = _norm_component(track.component)
    # "tail-light crack": detector may say lamp broken (or crack on a lamp) - equivalent
    if claimed in {"crack", "lamp broken"} and d in {"crack", "lamp broken"}:
        return d == "lamp broken" or comp in LAMP_COMPONENTS
    if claimed in {"crack", "glass shatter"} and d in {"crack", "glass shatter"} and comp in GLASS_COMPONENTS:
        return True
    return False


def _label(t: TrackEvidence) -> str:
    return f"{t.component or 'unlocated part'} {t.damage}".lower()


def verify(claim_text: str, tracks: list[TrackEvidence]) -> Verdict:
    claimed_comp, claimed_dmg = parse_claim(claim_text)
    base = dict(claimed_component=claimed_comp, claimed_damage=claimed_dmg)

    solid = [t for t in tracks if t.n_frames >= MIN_TRACK_FRAMES and not t.likely_reflection]
    summary = ", ".join(sorted({_label(t) for t in solid})) or "no persistent damage"

    if not claimed_dmg and not claimed_comp:
        return Verdict(verdict="UNCERTAIN", confidence=0.5, action="Physical Inspection",
                       reason="Claim text names no recognisable damage or component.",
                       detected_summary=summary, **base)
    if not solid:
        return Verdict(verdict="UNCERTAIN", confidence=0.6, action="Physical Inspection",
                       reason="Insufficient evidence: no damage persisted across enough frames.",
                       detected_summary="Insufficient evidence", **base)

    dmg_hits = [t for t in solid if damage_match(claimed_dmg, t)] if claimed_dmg else solid
    full_hits = ([t for t in dmg_hits if component_match(claimed_comp, _norm_component(t.component))]
                 if claimed_comp else dmg_hits)

    if full_hits:
        best = max(full_hits, key=lambda t: t.max_conf)
        conf = round(best.max_conf, 2)
        if best.max_conf >= STRONG_CONF:
            return Verdict(verdict="MATCH", confidence=conf, action="Remote Verification",
                           reason=f"Claimed damage confirmed on track #{best.track_id} over {best.n_frames} frames.",
                           detected_summary=summary, **base)
        return Verdict(verdict="UNCERTAIN", confidence=conf, action="Physical Inspection",
                       reason="Matching damage found but detector confidence is low.",
                       detected_summary=summary, **base)

    # Right damage type, but component cannot be confirmed (no parts model / unlocated).
    if dmg_hits and claimed_comp and not any(t.component for t in dmg_hits):
        best = max(dmg_hits, key=lambda t: t.max_conf)
        return Verdict(verdict="UNCERTAIN", confidence=round(min(best.max_conf, 0.75), 2),
                       action="Physical Inspection",
                       reason=f"{claimed_dmg} detected but its component could not be localised "
                              f"to the claimed {claimed_comp}.",
                       detected_summary=summary, **base)

    strong = [t for t in solid if t.max_conf >= STRONG_CONF]
    if strong:
        best = max(strong, key=lambda t: t.max_conf)
        claim_desc = " ".join(x for x in (claimed_comp, claimed_dmg) if x)
        return Verdict(verdict="MISMATCH", confidence=round(best.max_conf, 2), action="Manual Review",
                       reason=f"Claim says {claim_desc}, but detected: {summary}.",
                       detected_summary=summary, **base)
    best = max(solid, key=lambda t: t.max_conf)
    return Verdict(verdict="UNCERTAIN", confidence=round(best.max_conf, 2), action="Physical Inspection",
                   reason="Detected damage differs from the claim but with low confidence.",
                   detected_summary=summary, **base)


def inspect(tracks: list[TrackEvidence], frames_analyzed: int) -> Verdict:
    """Pre-insurance inspection: any persistent damage anywhere on the vehicle counts."""
    solid = [t for t in tracks if t.n_frames >= MIN_TRACK_FRAMES]
    real = [t for t in solid if not t.likely_reflection]
    suspect = [t for t in solid if t.likely_reflection]
    summary = ", ".join(sorted({_label(t) for t in real})) or "no damage detected"
    strong = [t for t in real if t.max_conf >= STRONG_CONF]
    if strong:
        best = max(strong, key=lambda t: t.max_conf)
        extra = f" ({len(suspect)} more filtered out as likely reflections)" if suspect else ""
        return Verdict(verdict="DAMAGED", confidence=round(best.max_conf, 2), action="Reject / Re-price",
                       reason=f"{len(real)} pre-existing damage region(s) found: {summary}.{extra}",
                       detected_summary=summary)
    if real:
        best = max(real, key=lambda t: t.max_conf)
        return Verdict(verdict="UNCERTAIN", confidence=round(best.max_conf, 2), action="Physical Inspection",
                       reason=f"Possible damage ({summary}) detected with low confidence; check in person.",
                       detected_summary=summary)
    if suspect:
        what = ", ".join(sorted({_label(t) for t in suspect}))
        return Verdict(verdict="UNCERTAIN", confidence=None, action="Re-capture Video",
                       reason=f"Detections ({what}) look like reflections or glare, not damage. Re-record in "
                              "shade or from another angle, or inspect in person.",
                       detected_summary="Possible reflections only")
    if frames_analyzed < MIN_COVERAGE_FRAMES:
        return Verdict(verdict="UNCERTAIN", confidence=None, action="Re-capture Video",
                       reason=f"Video too short ({frames_analyzed} frames analysed, need {MIN_COVERAGE_FRAMES}) "
                              "to certify the vehicle as undamaged.",
                       detected_summary="Insufficient coverage")
    return Verdict(verdict="CLEAN", confidence=None, action="Approve for Policy",
                   reason=f"No damage detected across {frames_analyzed} analysed frames.",
                   detected_summary="No damage detected")


def inspect_images(tracks: list[TrackEvidence], n_images: int) -> Verdict:
    """Pre-insurance inspection from still photos: no tracking, every detection counts."""
    real = [t for t in tracks if not t.likely_reflection]
    suspect = [t for t in tracks if t.likely_reflection]
    summary = ", ".join(sorted({t.damage for t in real})) or "no damage detected"
    strong = [t for t in real if t.max_conf >= STRONG_CONF]
    if strong:
        best = max(strong, key=lambda t: t.max_conf)
        extra = f" ({len(suspect)} more filtered out as likely reflections)" if suspect else ""
        return Verdict(verdict="DAMAGED", confidence=round(best.max_conf, 2), action="Reject / Re-price",
                       reason=f"{len(real)} damage region(s) found across {n_images} photo(s): {summary}.{extra}",
                       detected_summary=summary)
    if real:
        best = max(real, key=lambda t: t.max_conf)
        return Verdict(verdict="UNCERTAIN", confidence=round(best.max_conf, 2), action="Physical Inspection",
                       reason=f"Possible damage ({summary}) detected with low confidence; check in person.",
                       detected_summary=summary)
    if suspect:
        what = ", ".join(sorted({t.damage for t in suspect}))
        return Verdict(verdict="UNCERTAIN", confidence=None, action="Re-capture Photos",
                       reason=f"Detections ({what}) look like reflections or glare, not damage. Retake the photo "
                              "from another angle or in shade, or inspect in person.",
                       detected_summary="Possible reflections only")
    if n_images < MIN_PHOTOS:
        return Verdict(verdict="UNCERTAIN", confidence=None, action="Re-capture Photos",
                       reason=f"Only {n_images} photo(s); at least {MIN_PHOTOS} (front, rear, both sides) "
                              "are needed to certify the vehicle as undamaged.",
                       detected_summary="Insufficient coverage")
    return Verdict(verdict="CLEAN", confidence=None, action="Approve for Policy",
                   reason=f"No damage detected in {n_images} photos.", detected_summary="No damage detected")
