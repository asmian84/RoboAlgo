"""Ascending channel detector with multi-scale swing detection.

Detects upward-sloping parallel channels:
  - Both resistance (highs) and support (lows) trendlines slope upward
  - Lines are roughly parallel (slope difference < 45% of upper slope)
  - Channel height is at least 1.5% of mean price

Multi-scale scan: tries 4 different minimum_move scales so both
short-term (week) and long-term (month/year) channels are caught.

Visual output:
  - Two trendlines: upper resistance + lower support
  - Semi-transparent teal fill between the trendlines (fill_zone)
  - Breakout extension dashed line projected forward
"""

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

# Normalised slope thresholds
_MIN_SLOPE_RATIO = 0.0002   # must be meaningfully positive (not flat)
_MAX_SLOPE_RATIO = 0.06     # not an extreme ramp
_MIN_HEIGHT_RATIO = 0.015   # channel must be at least 1.5% wide
_MAX_SLOPE_GAP    = 0.45    # max slope diff as fraction of upper-line slope


def _detect_ascending(
    df: pd.DataFrame,
    highs: list,
    lows: list,
) -> dict[str, Any]:
    """Attempt ascending channel detection with the given pivot sets.

    Returns a result dict with confidence > 0 if a valid channel is found,
    or a NOT_PRESENT base_result otherwise.
    """
    result = base_result("Ascending Channel")
    if len(highs) < 2 or len(lows) < 2:
        return result

    last_idx   = len(df) - 1
    mean_price = max(float(df["close"].tail(60).mean()), 1.0)

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)

    h_slope, h_int = np.polyfit(xh, yh, 1)
    l_slope, l_int = np.polyfit(xl, yl, 1)

    # Both lines must slope upward
    if h_slope <= 0 or l_slope <= 0:
        return result

    # Slopes must be meaningful and not extreme
    if h_slope / mean_price < _MIN_SLOPE_RATIO:
        return result
    if h_slope / mean_price > _MAX_SLOPE_RATIO:
        return result

    # Lines must be roughly parallel — measure gap relative to the steeper line
    slope_gap = abs(h_slope - l_slope) / max(abs(h_slope), 1e-9)
    if slope_gap > _MAX_SLOPE_GAP:
        return result

    upper  = float(h_slope * last_idx + h_int)
    lower  = float(l_slope * last_idx + l_int)
    if upper <= lower:
        return result

    height = upper - lower
    if height / mean_price < _MIN_HEIGHT_RATIO:
        return result

    breakout     = upper
    invalidation = lower
    target       = float(breakout + height)

    structure_quality = float(np.clip(80 - slope_gap * 120, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    first_idx = int(min(xh[0], xl[0]))
    # Extension: project both lines 15 bars forward for the dashed breakout hint
    fwd_idx   = last_idx + 15
    overlay_lines = [
        # Upper resistance trendline
        [[first_idx, float(h_slope * first_idx + h_int)],
         [last_idx,  float(upper)]],
        # Lower support trendline
        [[first_idx, float(l_slope * first_idx + l_int)],
         [last_idx,  float(lower)]],
        # Breakout extension (dashed projection)
        [[last_idx,  float(upper)],
         [fwd_idx,   float(h_slope * fwd_idx + h_int)]],
    ]

    result.update({
        "pattern_name":       "Ascending Channel",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "probability":        round(probability, 2),
        "confidence":         round(probability, 2),
        "direction":          "bullish",
        "points": [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        # Teal fill between the two channel trendlines
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#14b8a6",
            "opacity": 0.22,
        },
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    """Multi-scale ascending channel detection.

    Scans pivot sets at 4 different sensitivity levels.  The channel with the
    highest composite probability (parallelism × volume × regime × momentum) wins.
    """
    result = base_result("Ascending Channel")
    if price_data is None or len(price_data) < 50:
        return result

    df           = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm  = compute_adaptive_minimum_move(df)

    best      = result
    best_prob = 0.0

    # Try progressively larger minimum-move scales to catch both
    # short-term (tight) and long-term (loose) channels.
    for scale in [1.0, 2.0, 3.0, 5.0]:
        mm     = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        highs  = swings["swing_highs"][-5:]
        lows   = swings["swing_lows"][-5:]
        r = _detect_ascending(df, highs, lows)
        if float(r.get("confidence", 0.0)) > best_prob:
            best_prob = float(r["confidence"])
            best      = r

    return best
