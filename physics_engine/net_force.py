"""Net force calculator: combines all force vectors into a single market force."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physics_engine.trend_force import compute_trend_force
from physics_engine.liquidity_force import compute_liquidity_force
from physics_engine.volatility_force import compute_volatility_force
from physics_engine.cycle_force import compute_cycle_force
from physics_engine.pattern_force import compute_pattern_force


# Default weights for combining forces
FORCE_WEIGHTS = {
    "trend": 0.30,
    "liquidity": 0.20,
    "volatility": 0.15,
    "cycle": 0.20,
    "pattern": 0.15,
}


def compute_net_force(
    df: pd.DataFrame,
    cycle_phase: float = 0.0,
    cycle_strength: float = 0.0,
    patterns: list[dict] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute net market force from all components.

    Returns dict with individual forces, net force, bias, and magnitude.
    """
    w = weights or FORCE_WEIGHTS

    trend = compute_trend_force(df)
    liquidity = compute_liquidity_force(df)
    volatility = compute_volatility_force(df)
    cycle = compute_cycle_force(cycle_phase, cycle_strength)
    pattern = compute_pattern_force(patterns or [])

    net = (
        trend * w.get("trend", 0.3)
        + liquidity * w.get("liquidity", 0.2)
        + volatility * w.get("volatility", 0.15)
        + cycle * w.get("cycle", 0.2)
        + pattern * w.get("pattern", 0.15)
    )
    net = float(np.clip(net, -1, 1))

    if net > 0.1:
        bias = "bullish"
    elif net < -0.1:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "trend_force": trend,
        "liquidity_force": liquidity,
        "volatility_force": volatility,
        "cycle_force": cycle,
        "pattern_force": pattern,
        "net_force": round(net, 4),
        "bias": bias,
        "force_magnitude": round(abs(net), 4),
    }
