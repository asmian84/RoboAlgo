"""Bullish and Bearish Pennant detectors.

Pennant = short, tight symmetrical triangle forming after a sharp pole.

Bullish Pennant:
  - Sharp upward pole
  - Converging highs (falling) + rising lows → symmetrical triangle consolidation
  - Breakout above upper boundary → target = breakout + pole_height

Bearish Pennant:
  - Sharp downward pole
  - Rising highs (counter-trend) + falling lows → symmetrical triangle
  - Breakdown below lower boundary → target = breakdown − pole_height
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


def _pennant_body(
    df: pd.DataFrame,
    post_start_global: int,
    pole_height: float,
    bullish: bool,
    adaptive_mm: float,
) -> dict[str, Any] | None:
    """Detect pennant triangle body after a pole. Returns partial dict or None."""
    post = df.iloc[post_start_global:].reset_index()
    if len(post) < 8:
        return None

    swings = detect_swings(post.rename(columns={"index": "orig_index"}), minimum_move=adaptive_mm)
    highs = swings["swing_highs"][-4:]
    lows  = swings["swing_lows"][-4:]
    if len(highs) < 2 or len(lows) < 2:
        return None

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)
    h_slope, h_int = np.polyfit(xh, yh, 1)
    l_slope, l_int = np.polyfit(xl, yl, 1)

    if bullish:
        # Pennant after bullish pole: highs slope down, lows slope up → converging
        if not (h_slope < 0 and l_slope > 0):
            return None
    else:
        # Pennant after bearish pole: highs slope up, lows slope down → converging
        if not (h_slope > 0 and l_slope < 0):
            return None

    last_idx   = len(df) - 1
    local_last = int(last_idx - post["index"].iloc[0])
    upper      = float(h_slope * local_last + h_int)
    lower      = float(l_slope * local_last + l_int)

    width_now  = max(upper - lower, 0.0)
    width_hist = float(np.mean(yh) - np.mean(yl))

    # Pennant must be compact relative to the pole
    if width_hist <= 0 or width_now >= width_hist * 0.9 or width_now > pole_height * 0.5:
        return None

    first_idx = int(min(xh[0], xl[0]))
    # Convert local (post sub-DF, 0-based) indices → global bar indices
    # post_start_global is the global offset of post's first row
    post_global_offset = int(post["index"].iloc[0])
    g_first = post_global_offset + first_idx
    g_last  = len(df) - 1   # == post_global_offset + local_last
    overlay_lines = [
        [[g_first, float(h_slope * first_idx + h_int)], [g_last, upper]],
        [[g_first, float(l_slope * first_idx + l_int)], [g_last, lower]],
    ]

    if bullish:
        breakout     = upper
        invalidation = lower
        target       = upper + pole_height
    else:
        breakout     = lower
        invalidation = upper
        target       = lower - pole_height

    return {
        "breakout": breakout, "invalidation": invalidation, "target": target,
        "overlay_lines": overlay_lines,
        "highs": highs, "lows": lows,
        "post_global_offset": post_global_offset,
        "width_now": width_now, "width_hist": width_hist,
    }


def _detect_bullish_pennant(df: pd.DataFrame, adaptive_mm: float) -> dict[str, Any]:
    result = base_result("Bullish Pennant")
    close = df["close"].astype(float)
    pole_window = close.tail(35).reset_index(drop=True)
    low_idx  = int(pole_window.idxmin())
    high_idx = int(pole_window.idxmax())
    if high_idx <= low_idx:
        return result

    pole_move = (float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])) / max(
        float(pole_window.iloc[low_idx]), 1e-9
    )
    min_pole = max(3.0 * adaptive_mm, 0.03)
    if pole_move < min_pole:
        return result

    pole_height  = float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])
    global_start = len(close) - len(pole_window) + high_idx
    body = _pennant_body(df, global_start, pole_height, bullish=True, adaptive_mm=adaptive_mm)
    if body is None:
        return result

    breakout, invalidation, target = body["breakout"], body["invalidation"], body["target"]
    sq = float(np.clip(
        70 + pole_move * 100 - (body["width_now"] / max(body["width_hist"], 1e-9)) * 30, 0, 100
    ))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    # Pole: bottom of impulse → top of impulse (global indices)
    pole_b_global = len(close) - len(pole_window) + low_idx
    pole_t_global = global_start
    pole_seg = [[pole_b_global, float(pole_window.iloc[low_idx])],
                [pole_t_global, float(pole_window.iloc[high_idx])]]

    g_off = body["post_global_offset"]
    result.update({
        "pattern_name": "Bullish Pennant", "status": status,
        "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4), "confidence": round(prob, 2),
        "probability": round(prob, 2), "direction": "bullish",
        "points": [[g_off + p["index"], p["price"]] for p in body["highs"][-2:] + body["lows"][-2:]],
        "overlay_lines": [pole_seg] + body["overlay_lines"],
        "overlay_line_roles": ["pole", "resistance", "support"],
    })
    return result


def _detect_bearish_pennant(df: pd.DataFrame, adaptive_mm: float) -> dict[str, Any]:
    result = base_result("Bearish Pennant")
    close = df["close"].astype(float)
    pole_window = close.tail(35).reset_index(drop=True)
    high_idx = int(pole_window.idxmax())
    low_idx  = int(pole_window.idxmin())
    if low_idx <= high_idx:
        return result

    pole_move = (float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])) / max(
        float(pole_window.iloc[high_idx]), 1e-9
    )
    min_pole = max(3.0 * adaptive_mm, 0.03)
    if pole_move < min_pole:
        return result

    pole_height  = float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])
    global_start = len(close) - len(pole_window) + low_idx
    body = _pennant_body(df, global_start, pole_height, bullish=False, adaptive_mm=adaptive_mm)
    if body is None:
        return result

    breakout, invalidation, target = body["breakout"], body["invalidation"], body["target"]
    sq = float(np.clip(
        70 + pole_move * 100 - (body["width_now"] / max(body["width_hist"], 1e-9)) * 30, 0, 100
    ))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    # Pole: top of impulse → bottom of impulse (global indices)
    pole_t_global = len(close) - len(pole_window) + high_idx
    pole_b_global = global_start
    pole_seg = [[pole_t_global, float(pole_window.iloc[high_idx])],
                [pole_b_global, float(pole_window.iloc[low_idx])]]

    g_off = body["post_global_offset"]
    result.update({
        "pattern_name": "Bearish Pennant", "status": status,
        "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4), "confidence": round(prob, 2),
        "probability": round(prob, 2), "direction": "bearish",
        "points": [[g_off + p["index"], p["price"]] for p in body["highs"][-2:] + body["lows"][-2:]],
        "overlay_lines": [pole_seg] + body["overlay_lines"],
        "overlay_line_roles": ["pole", "resistance", "support"],
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Bullish Pennant, Bearish Pennant] detections."""
    if price_data is None or len(price_data) < 60:
        return [base_result("Bullish Pennant"), base_result("Bearish Pennant")]

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    return [
        _detect_bullish_pennant(df, adaptive_mm),
        _detect_bearish_pennant(df, adaptive_mm),
    ]
