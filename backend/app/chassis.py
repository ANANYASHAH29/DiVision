"""Chassis number (VIN) normalisation and validation."""
import re

# ISO 3779: 17 characters, letters and digits, never I, O or Q.
_VIN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def normalize(raw: str) -> str:
    """Return the upper-cased chassis number or raise ValueError with a user-facing message."""
    vin = re.sub(r"[\s-]", "", (raw or "")).upper()
    if not vin:
        raise ValueError("Chassis number is required.")
    if len(vin) != 17:
        raise ValueError(f"Chassis number must be 17 characters (got {len(vin)}).")
    if not _VIN.match(vin):
        raise ValueError("Chassis number may only contain letters and digits, and never I, O or Q.")
    return vin
