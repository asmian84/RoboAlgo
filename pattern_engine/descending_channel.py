"""Descending Channel (Falling Channel) detector with multi-scale swing detection.

A descending channel has:
  - Both trendlines sloping downward (negative slope)
  - Resistance trendline through swing highs
  - Support trendline through swing lows
  - Roughly parallel lines (similar slopes)

Multi-scale scan: tries 4 different minimum_move scales so both
short-term (week) and long-term (month/year) channels are caught.

Trading implications:
  - Bearish trend continuation while inside channel
  - Bullish breakout above resistance = reversal signal
  - Target = breakout + channel width

Visual output:
  - Two trendlines: upper resistance + lower support
  - Semi-transparent red fill between the trendlines (fill_zone)
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
from structure_engine.swing_detector import compute_adaptive_minimum_move, detect_swings

_MIN_SLOPE_RATIO = 0.0002
_MAX_SLOPE_RATIO = 0.06
_MIN_HEIGHT_RATIO = 0.015
_MAX_SLOPE_GAP    = 0.45    # max slope diff as fraction of upper-line slope


def _detect_descending(
    df: pd.DataFrame,
    highs: list,
    lows: list,
) -> dict[str, Any]:
    """Attempt descending channel detection with the given pivot sets."""
    result = base_result("Descending Channel")
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

    # Both lines must slope downward
    if h_slope >= 0 or l_slope >= 0:
        return result

    # Slopes must be meaningful and not extreme
    if abs(h_slope) / mean_price < _MIN_SLOPE_RATIO:
        return result
    if abs(h_slope) / mean_price > _MAX_SLOPE_RATIO:
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

    # Bullish breakout above resistance (reversal of the downtrend)
    breakout     = upper
    invalidation = lower
    target       = float(breakout + height)

    sq   = float(np.clip(80 - slope_gap * 120, 0, 100))
    prob = composite_probability(
        sq, volume_confirmation(df),
        liquidity_alignment_score(df, breakout),
        market_regime_score(df), momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    first_idx = int(min(xh[0], xl[0]))
    fwd_idx   = last_idx + 15
    overlay_lines = [
        [[first_idx, float(h_slope * first_idx + h_int)], [last_idx, float(upper)]],
        [[first_idx, float(l_slope * first_idx + l_int)], [last_idx, float(lower)]],
        # Breakout extension projection
        [[last_idx, float(upper)], [fwd_idx, float(h_slope * fwd_idx + h_int)]],
    ]

    result.update({
        "pattern_name":       "Descending Channel",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "bullish",   # breakout direction = bullish reversal
        "points": [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        # Muted red fill between the two channel trendlines
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#f87171",
            "opacity": 0.20,
        },
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    """Multi-scale descending channel detection."""
    result = base_result("Descending Channel")
    if price_data is None or len(price_data) < 50:
        return result

    df          = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)

    best      = result
    best_prob = 0.0

    for scale in [1.0, 2.0, 3.0, 5.0]:
        mm     = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        highs  = swings["swing_highs"][-5:]
        lows   = swings["swing_lows"][-5:]
        r = _detect_descending(df, highs, lows)
        if float(r.get("confidence", 0.0)) > best_prob:
            best_prob = float(r["confidence"])
            best      = r

    return best
