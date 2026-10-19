from app.schemas import TrackEvidence
from app.verification import parse_claim, verify


def T(dmg, comp, conf, n=20):
    return TrackEvidence(track_id=1, damage=dmg, component=comp, max_conf=conf, mean_conf=conf,
                         n_frames=n, first_frame=0, best_frame=1)


def test_parse():
    assert parse_claim("Rear bumper scratch") == ("rear bumper", "scratch")
    assert parse_claim("Tail-light crack") == ("tail light", "crack")


def test_mismatch_slide_example():
    v = verify("Rear bumper scratch", [T("dent", "door", 0.92)])
    assert v.verdict == "MISMATCH" and v.action == "Manual Review"


def test_match():
    assert verify("Front bonnet dent", [T("dent", "bonnet", 0.97)]).verdict == "MATCH"


def test_lamp_crack_equivalence():
    assert verify("Tail-light crack", [T("lamp broken", "tail light", 0.95)]).verdict == "MATCH"


def test_uncertain_insufficient():
    v = verify("Door scratch", [T("scratch", "door", 0.61, n=2)])
    assert v.verdict == "UNCERTAIN" and v.action == "Physical Inspection"


def test_no_parts_model_is_uncertain_not_mismatch():
    assert verify("Rear bumper scratch", [T("scratch", None, 0.9)]).verdict == "UNCERTAIN"


def test_chassis_validation():
    import pytest
    from app.chassis import normalize
    assert normalize("ma3 ewde1s00-123456") == "MA3EWDE1S00123456"
    for bad in ("", "ABC", "MA3EWDE1S0012345O", "MA3EWDE1S001234567"):
        with pytest.raises(ValueError):
            normalize(bad)


def test_reflection_assess():
    from app.reflection import assess
    # verifier alone is enough
    assert assess(["glass shatter"] * 5, list(range(5)), real_prob=0.1, verifier_threshold=0.4)["likely"]
    assert not assess(["glass shatter"] * 5, list(range(5)), real_prob=0.9, verifier_threshold=0.4)["likely"]
    # one temporal sign is not enough, two are
    flip = ["glass shatter", "crack", "scratch", "glass shatter"]
    assert not assess(flip, [0, 1, 2, 3])["likely"]
    assert assess(flip, [0, 5, 9, 14])["likely"]
    # no verifier installed -> no verifier opinion
    assert not assess(["dent"] * 4, [0, 1, 2, 3], real_prob=None, verifier_threshold=None)["likely"]


def test_reflection_never_gives_clean_or_damaged():
    from app.verification import inspect, inspect_images
    refl = T("glass shatter", None, 0.92).model_copy(update=dict(likely_reflection=True))
    v = inspect([refl], 200)
    assert v.verdict == "UNCERTAIN" and v.action == "Re-capture Video"
    assert inspect_images([refl], 6).verdict == "UNCERTAIN"
    # a real strong damage alongside a reflection still gives DAMAGED, and mentions the filtered one
    real = T("dent", None, 0.8)
    v = inspect([real, refl], 200)
    assert v.verdict == "DAMAGED" and "reflection" in v.reason


def test_inspect_clean_damaged_uncertain():
    from app.verification import inspect
    assert inspect([], 200).verdict == "CLEAN"
    assert inspect([], 10).verdict == "UNCERTAIN"
    assert inspect([T("scratch", None, 0.8)], 200).verdict == "DAMAGED"
    assert inspect([T("scratch", None, 0.3)], 200).verdict == "UNCERTAIN"
    assert inspect([T("scratch", None, 0.9, n=1)], 200).verdict == "CLEAN"  # one-frame blip ignored
