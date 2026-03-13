"""Megaphone/expansion detector using swing anchors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pattern_engine.common import (
    base_result,
    composite_probability,
    liquidity_alignment_score,
    market_regime_score,
    momentum_score,
    status_from_levels,
    volume_confirmation,
)
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    result = base_result("Megaphone")
    if price_data is None or len(price_data) < 70:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs = swings["swing_highs"][-5:]
    lows = swings["swing_lows"][-5:]
    if len(highs) < 3 or len(lows) < 3:
        return result

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows], dtype=float)
    yl = np.array([p["price"] for p in lows], dtype=float)
    h_slope, h_int = np.polyfit(xh, yh, 1)
    l_slope, l_int = np.polyfit(xl, yl, 1)
    # Megaphone: expanding — highs rising, lows falling
    if h_slope <= 0 or l_slope >= 0:
        return result

    last_idx = len(df) - 1
    upper = float(h_slope * last_idx + h_int)
    lower = float(l_slope * last_idx + l_int)
    width_now = max(upper - lower, 0.0)
    early_idx = max(int(np.mean([p["index"] for p in highs[:2] + lows[:2]])), 1)
    early_upper = float(h_slope * early_idx + h_int)
    early_lower = float(l_slope * early_idx + l_int)
    width_early = max(early_upper - early_lower, 1e-9)
    if width_now <= width_early * 1.2:
        return result

    breakout = upper
    invalidation = lower
    target = float(breakout + width_now * 0.75)
    structure_quality = float(np.clip(55 + (width_now / width_early) * 20, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    first_idx = int(min(xh[0], xl[0]))
    overlay_lines = [
        [[first_idx, float(h_slope * first_idx + h_int)], [last_idx, upper]],
        [[first_idx, float(l_slope * first_idx + l_int)], [last_idx, lower]],
    ]

    result.update({
        "pattern_name":       "Megaphone",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(probability, 2),
        "probability":        round(probability, 2),
        "direction":          "neutral",
        "points":             [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support"],
    })
    return result

