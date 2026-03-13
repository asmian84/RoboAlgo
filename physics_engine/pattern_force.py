"""Pattern force: directional pressure from detected chart/harmonic/wyckoff patterns."""

from __future__ import annotations

import numpy as np


def compute_pattern_force(patterns: list[dict]) -> float:
    """Compute pattern force from -1 (strong bearish patterns) to +1 (strong bullish patterns).

    Weights each active pattern by its confidence and direction.
    """
    if not patterns:
        return 0.0

    bull_force = 0.0
    bear_force = 0.0
    total_weight = 0.0

    for p in patterns:
        status = p.get("status", "NOT_PRESENT")
        if status in ("NOT_PRESENT", "FAILED"):
            continue

        conf = float(p.get("confidence", 0.0)) / 100.0  # normalize to 0-1
        direction = p.get("direction", "neutral")

        # Weight by status urgency
        status_weight = {"BREAKOUT": 1.5, "READY": 1.2, "COMPLETED": 0.8, "FORMING": 0.6}.get(status, 0.5)

        weight = conf * status_weight
        total_weight += weight

        if direction == "bullish":
            bull_force += weight
        elif direction == "bearish":
            bear_force += weight

    if total_weight <= 0:
        return 0.0

    force = (bull_force - bear_force) / total_weight
    return round(float(np.clip(force, -1, 1)), 4)
