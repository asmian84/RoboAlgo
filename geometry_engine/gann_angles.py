"""Gann angle computation — normalized price-per-bar slopes for all 9 angles."""

from __future__ import annotations

import numpy as np

# Canonical Gann angles: (name, price_units_per_time_unit_multiplier)
GANN_ANGLE_DEFS: list[tuple[str, float]] = [
    ("8x1", 8.0),
    ("4x1", 4.0),
    ("3x1", 3.0),
    ("2x1", 2.0),
    ("1x1", 1.0),
    ("1x2", 0.5),
    ("1x3", 1.0 / 3.0),
    ("1x4", 0.25),
    ("1x8", 0.125),
]


def compute_gann_angles(
    anchor_price: float,
    anchor_idx: int,
    pivot_price: float,
    pivot_idx: int,
) -> dict:
    """Compute all 9 Gann fan lines from an anchor-pivot pair.

    Returns dict with slope_1x1 and list of angle projections.
    """
    bars = max(pivot_idx - anchor_idx, 1)
    rise = pivot_price - anchor_price
    if rise <= 0:
        return {"slope_1x1": 0.0, "angles": []}

    slope_1x1 = rise / bars

    angles = []
    for name, mult in GANN_ANGLE_DEFS:
        angles.append({
            "name": name,
            "multiplier": mult,
            "slope": round(slope_1x1 * mult, 6),
        })

    return {
        "slope_1x1": round(slope_1x1, 6),
        "angles": angles,
    }


def project_fan_price(
    anchor_price: float,
    anchor_idx: int,
    slope_1x1: float,
    target_idx: int,
    multiplier: float = 1.0,
) -> float:
    """Project price at target_idx along a Gann fan line."""
    dt = target_idx - anchor_idx
    return anchor_price + slope_1x1 * multiplier * dt
