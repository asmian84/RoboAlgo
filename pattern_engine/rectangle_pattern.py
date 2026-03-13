"""Rectangle / Trading Range pattern detector.

A rectangle forms when price oscillates between two roughly horizontal levels:
  - Resistance: flat upper boundary (tested 2+ times)
  - Support: flat lower boundary (tested 2+ times)

The breakout direction determines bullish/bearish bias:
  - Bullish breakout above resistance → target = resistance + height
  - Bearish breakdown below support   → target = support − height
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

_FLAT_RATIO = 0.0015   # slope/price ratio below which a trendline is "flat"


def _is_flat(slope: float, mean_price: float) -> bool:
    if mean_price <= 0:
        return abs(slope) < _FLAT_RATIO
    return abs(slope / mean_price) < _FLAT_RATIO


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    result = base_result("Rectangle")
    if price_data is None or len(price_data) < 60:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs = swings["swing_highs"][-6:]
    lows  = swings["swing_lows"][-6:]
    if len(highs) < 2 or len(lows) < 2:
        return result

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)
    h_slope, h_int = np.polyfit(xh, yh, 1)
    l_slope, l_int = np.polyfit(xl, yl, 1)

    mean_price = float(np.mean(np.concatenate([yh, yl])))
    if not (_is_flat(h_slope, mean_price) and _is_flat(l_slope, mean_price)):
        return result

    last_idx   = len(df) - 1
    resistance = float(h_slope * last_idx + h_int)
    support    = float(l_slope * last_idx + l_int)
    height     = resistance - support

    # Need at least 2% height and positive gap
    if height <= 0 or height / max(mean_price, 1e-9) < 0.015:
        return result

    # Determine which boundary price is near → bias breakout direction
    last_close = float(df["close"].iloc[-1])
    if last_close >= resistance * 0.995:
        breakout, target, invalidation, bullish = resistance, resistance + height, support, True
    elif last_close <= support * 1.005:
        breakout, target, invalidation, bullish = support, support - height, resistance, False
    else:
        # Mid-range: bias bullish (statistically more common outcome)
        breakout, target, invalidation, bullish = resistance, resistance + height, support, True

    sq = float(np.clip(
        72 - (abs(h_slope) + abs(l_slope)) / max(mean_price, 1e-9) * 4000, 0, 100
    ))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=bullish)

    first_idx = int(min(xh[0], xl[0]))
    overlay_lines = [
        [[first_idx, float(h_slope * first_idx + h_int)], [last_idx, resistance]],
        [[first_idx, float(l_slope * first_idx + l_int)], [last_idx, support]],
    ]

    result.update({
        "pattern_name":       "Rectangle",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "bullish" if bullish else "bearish",
        "points":             [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support"],
        "support_level":      round(support, 4),
        "resistance_level":   round(resistance, 4),
    })
    return result
