"""Rising and Falling Wedge pattern detector.

Rising Wedge (bearish):
  - Two converging trendlines both sloping upward
  - Lower trendline slope > upper trendline slope → apex forms above/right
  - Breakdown below lower trendline → target = entry − height

Falling Wedge (bullish):
  - Two converging trendlines both sloping downward
  - Upper trendline slope > lower trendline slope (less negative) → apex above/right
  - Breakout above upper trendline → target = entry + height

Both require at least 2 swing highs and 2 swing lows within the wedge span.
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

# Minimum normalised slope to qualify as genuinely sloping (not flat)
_MIN_SLOPE_RATIO = 0.0003
# Maximum normalised slope (extreme slopes → noise / insufficient data)
_MAX_SLOPE_RATIO = 0.08
# Minimum wedge height (% of mean price) to bother
_MIN_HEIGHT_RATIO = 0.012


def _fit(points: list[dict]) -> tuple[float, float]:
    """OLS trendline through pivot points → (slope, intercept)."""
    x = np.array([p["index"] for p in points], dtype=float)
    y = np.array([p["price"] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _detect_rising_wedge(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Detect Rising Wedge (bearish reversal/continuation)."""
    result = base_result("Rising Wedge")
    if len(highs) < 2 or len(lows) < 2:
        return result

    last_idx = len(df) - 1
    mean_price = max(float(df["close"].tail(60).mean()), 1.0)

    # Use last 5 pivots at each side
    rh = highs[-5:]
    rl = lows[-5:]
    if len(rh) < 2 or len(rl) < 2:
        return result

    h_slope, h_int = _fit(rh)
    l_slope, l_int = _fit(rl)

    # Both lines must slope upward
    if h_slope <= 0 or l_slope <= 0:
        return result

    # Lower line rises faster → lines converge toward an apex above
    if l_slope <= h_slope:
        return result

    # Slopes must be meaningful (not flat, not extreme)
    if abs(l_slope) / mean_price < _MIN_SLOPE_RATIO:
        return result
    if abs(h_slope) / mean_price > _MAX_SLOPE_RATIO:
        return result

    resistance = h_slope * last_idx + h_int
    support    = l_slope * last_idx + l_int
    if resistance <= support:
        return result

    height = resistance - support
    if height / mean_price < _MIN_HEIGHT_RATIO:
        return result

    # Wedge span: from oldest pivot to last bar
    first_idx = int(min(rh[0]["index"], rl[0]["index"]))

    breakout    = float(support)
    target      = float(support - height)
    invalidation = float(resistance)

    # Score: tighter convergence = higher quality
    conv = float(np.clip((l_slope - h_slope) / max(h_slope, 1e-9) * 40, 0, 30))
    sq   = float(np.clip(65 + conv, 0, 100))
    prob = composite_probability(
        sq, volume_confirmation(df),
        liquidity_alignment_score(df, breakout),
        market_regime_score(df), momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    overlay_lines = [
        # Upper resistance trendline
        [[first_idx, float(h_slope * first_idx + h_int)], [last_idx, float(resistance)]],
        # Lower support trendline
        [[first_idx, float(l_slope * first_idx + l_int)], [last_idx, float(support)]],
        # Breakdown extension (dashed)
        [[last_idx, float(support)], [last_idx + 20, float(l_slope * (last_idx + 20) + l_int)]],
    ]

    result.update({
        "pattern_name":       "Rising Wedge",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "bearish",
        "points": [
            [int(rh[0]["index"]),  float(rh[0]["price"])],
            [int(rh[-1]["index"]), float(rh[-1]["price"])],
            [int(rl[0]["index"]),  float(rl[0]["price"])],
            [int(rl[-1]["index"]), float(rl[-1]["price"])],
        ],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        # Red fill — bearish squeeze
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#ef4444",
            "opacity": 0.20,
        },
    })
    return result


def _detect_falling_wedge(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Detect Falling Wedge (bullish reversal/continuation)."""
    result = base_result("Falling Wedge")
    if len(highs) < 2 or len(lows) < 2:
        return result

    last_idx = len(df) - 1
    mean_price = max(float(df["close"].tail(60).mean()), 1.0)

    rh = highs[-5:]
    rl = lows[-5:]
    if len(rh) < 2 or len(rl) < 2:
        return result

    h_slope, h_int = _fit(rh)
    l_slope, l_int = _fit(rl)

    # Both lines must slope downward
    if h_slope >= 0 or l_slope >= 0:
        return result

    # Upper line falls slower than lower → lines converge toward apex below/right
    if h_slope <= l_slope:
        return result

    if abs(h_slope) / mean_price < _MIN_SLOPE_RATIO:
        return result
    if abs(l_slope) / mean_price > _MAX_SLOPE_RATIO:
        return result

    resistance = h_slope * last_idx + h_int
    support    = l_slope * last_idx + l_int
    if resistance <= support:
        return result

    height = resistance - support
    if height / mean_price < _MIN_HEIGHT_RATIO:
        return result

    first_idx = int(min(rh[0]["index"], rl[0]["index"]))

    breakout    = float(resistance)
    target      = float(resistance + height)
    invalidation = float(support)

    conv = float(np.clip((h_slope - l_slope) / max(abs(l_slope), 1e-9) * 40, 0, 30))
    sq   = float(np.clip(65 + conv, 0, 100))
    prob = composite_probability(
        sq, volume_confirmation(df),
        liquidity_alignment_score(df, breakout),
        market_regime_score(df), momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    overlay_lines = [
        # Upper resistance trendline
        [[first_idx, float(h_slope * first_idx + h_int)], [last_idx, float(resistance)]],
        # Lower support trendline
        [[first_idx, float(l_slope * first_idx + l_int)], [last_idx, float(support)]],
        # Breakout extension (dashed)
        [[last_idx, float(resistance)], [last_idx + 20, float(h_slope * (last_idx + 20) + h_int)]],
    ]

    result.update({
        "pattern_name":       "Falling Wedge",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "bullish",
        "points": [
            [int(rh[0]["index"]),  float(rh[0]["price"])],
            [int(rh[-1]["index"]), float(rh[-1]["price"])],
            [int(rl[0]["index"]),  float(rl[0]["price"])],
            [int(rl[-1]["index"]), float(rl[-1]["price"])],
        ],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        # Green/teal fill — bullish compression
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#22c55e",
            "opacity": 0.20,
        },
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Rising Wedge, Falling Wedge] detections."""
    if price_data is None or len(price_data) < 50:
        return [base_result("Rising Wedge"), base_result("Falling Wedge")]

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs  = swings["swing_highs"]
    lows   = swings["swing_lows"]

    return [
        _detect_rising_wedge(df, highs, lows),
        _detect_falling_wedge(df, highs, lows),
    ]
