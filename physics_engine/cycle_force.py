"""Cycle force: directional momentum from dominant cycle phase."""

from __future__ import annotations

import math

import numpy as np


def compute_cycle_force(cycle_phase: float, cycle_strength: float) -> float:
    """Compute cycle force from -1 (cycle trough approaching) to +1 (cycle peak approaching).

    Uses cos(2π * phase) scaled by cycle strength.
    Phase 0.25 = peak (force = 0, about to decline) → cycle_force going negative
    Phase 0.75 = trough (force = 0, about to rise) → cycle_force going positive

    The DERIVATIVE of cosine gives the directional force:
    force = -sin(2π * phase) * strength
    """
    if cycle_strength <= 0 or cycle_phase < 0:
        return 0.0

    # Derivative of cos gives direction of cycle movement
    # -sin at phase 0 = 0 (neutral), phase 0.25 = -1 (declining), phase 0.75 = +1 (rising)
    force = -math.sin(2 * math.pi * cycle_phase) * min(cycle_strength, 1.0)

    return round(float(np.clip(force, -1, 1)), 4)
