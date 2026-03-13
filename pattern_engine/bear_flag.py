"""Bear Flag detector (bearish continuation).

Bear Flag:
  - Sharp downward pole (price drops significantly)
  - Consolidation that drifts UPWARD (parallel channel with positive slope)
  - Breakdown below lower channel boundary = bearish signal
  - Target = breakdown − pole_height
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


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    result = base_result("Bear Flag")
    if price_data is None or len(price_data) < 60:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    close = df["close"].astype(float)

    # Look for pole in the last 35 bars
    pole_window = close.tail(35).reset_index(drop=True)
    high_idx = int(pole_window.idxmax())
    low_idx  = int(pole_window.idxmin())

    # Bearish pole: high must come BEFORE low
    if low_idx <= high_idx:
        return result

    pole_move = (float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])) / max(
        float(pole_window.iloc[high_idx]), 1e-9
    )
    min_pole = max(3.0 * adaptive_mm, 0.03)
    if pole_move < min_pole or low_idx - high_idx > 16:
        return result

    pole_height = float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])

    # Consolidation phase starts at the pole low
    global_start = len(close) - len(pole_window) + low_idx
    post = df.iloc[global_start:].reset_index()
    if len(post) < 6:
        return result

    swings = detect_swings(post.rename(columns={"index": "orig_index"}), minimum_move=adaptive_mm)
    highs = swings["swing_highs"][-4:]
    lows  = swings["swing_lows"][-4:]
    if len(highs) < 2 or len(lows) < 2:
        return result

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)
    h_slope, h_int = np.polyfit(xh, yh, 1)
    l_slope, l_int = np.polyfit(xl, yl, 1)

    # Bear flag: both channel lines slope UPWARD (counter-trend drift)
    if not (h_slope > 0 and l_slope > 0):
        return result

    # Channel width should be tighter than the pole
    last_idx   = len(df) - 1
    local_last = int(last_idx - post["index"].iloc[0])
    upper_now  = float(h_slope * local_last + h_int)
    lower_now  = float(l_slope * local_last + l_int)
    channel_w  = max(upper_now - lower_now, 0.0)
    if channel_w >= pole_height * 0.5:
        return result

    breakout     = lower_now          # bearish: break BELOW lower channel
    invalidation = upper_now          # price above upper = flag failed
    target       = float(breakout - pole_height)

    structure_quality = float(np.clip(65 + pole_move * 150 - abs(h_slope - l_slope) * 30, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    first_idx = int(min(xh[0], xl[0]))
    # Convert local (post sub-DF, 0-based) indices → global bar indices
    g_first = global_start + first_idx
    # Pole: from pole top → pole bottom (global bar indices)
    pole_top_global    = len(close) - len(pole_window) + high_idx
    pole_bottom_global = global_start   # == len(close) - len(pole_window) + low_idx
    pole_top_price     = float(pole_window.iloc[high_idx])
    pole_bottom_price  = float(pole_window.iloc[low_idx])

    overlay_lines = [
        # Flagpole (bearish: price drops sharply)
        [[pole_top_global, pole_top_price], [pole_bottom_global, pole_bottom_price]],
        # Flag channel: upper trendline
        [[g_first, float(h_slope * first_idx + h_int)], [last_idx, upper_now]],
        # Flag channel: lower trendline
        [[g_first, float(l_slope * first_idx + l_int)], [last_idx, lower_now]],
    ]

    result.update({
        "pattern_name":       "Bear Flag",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(probability, 2),
        "probability":        round(probability, 2),
        "direction":          "bearish",
        "points": [[global_start + p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["pole", "resistance", "support"],
    })
    return result
