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

    # ── Draw lines through ACTUAL swing pivots, extended forward ──────
    fwd_idx = last_idx + 15
    # Upper line: first swing high → last swing high → extended forward
    h0_idx, h0_p = int(rh[0]["index"]), float(rh[0]["price"])
    h1_idx, h1_p = int(rh[-1]["index"]), float(rh[-1]["price"])
    h_run = max(h1_idx - h0_idx, 1)
    h_sl  = (h1_p - h0_p) / h_run
    h_fwd = h1_p + h_sl * (fwd_idx - h1_idx)

    # Lower line: first swing low → last swing low → extended forward
    l0_idx, l0_p = int(rl[0]["index"]), float(rl[0]["price"])
    l1_idx, l1_p = int(rl[-1]["index"]), float(rl[-1]["price"])
    l_run = max(l1_idx - l0_idx, 1)
    l_sl  = (l1_p - l0_p) / l_run
    l_fwd = l1_p + l_sl * (fwd_idx - l1_idx)

    overlay_lines = [
        # Upper resistance: through actual swing highs, extended forward
        [[h0_idx, h0_p], [fwd_idx, round(h_fwd, 4)]],
        # Lower support: through actual swing lows, extended forward
        [[l0_idx, l0_p], [fwd_idx, round(l_fwd, 4)]],
        # Measured move: breakout → target (shows expected drop)
        [[last_idx, round(breakout, 4)], [last_idx, round(target, 4)]],
    ]

    # Points: all swing pivots for circle markers
    all_pts = [[int(p["index"]), float(p["price"])] for p in rh] + \
              [[int(p["index"]), float(p["price"])] for p in rl]
    all_pts.sort(key=lambda x: x[0])
    pt_labels = [f"H{i+1}" if any(p["index"] == pt[0] and p["price"] == pt[1] for p in rh)
                 else f"L{i+1}" for i, pt in enumerate(all_pts)]

    result.update({
        "pattern_name":       "Rising Wedge",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "bearish",
        "points":             all_pts,
        "point_labels":       pt_labels,
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "measured_move"],
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#ef4444",
            "opacity": 0.10,
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

    # ── Draw lines through ACTUAL swing pivots, extended forward ──────
    fwd_idx = last_idx + 15
    h0_idx, h0_p = int(rh[0]["index"]), float(rh[0]["price"])
    h1_idx, h1_p = int(rh[-1]["index"]), float(rh[-1]["price"])
    h_run = max(h1_idx - h0_idx, 1)
    h_sl  = (h1_p - h0_p) / h_run
    h_fwd = h1_p + h_sl * (fwd_idx - h1_idx)

    l0_idx, l0_p = int(rl[0]["index"]), float(rl[0]["price"])
    l1_idx, l1_p = int(rl[-1]["index"]), float(rl[-1]["price"])
    l_run = max(l1_idx - l0_idx, 1)
    l_sl  = (l1_p - l0_p) / l_run
    l_fwd = l1_p + l_sl * (fwd_idx - l1_idx)

    overlay_lines = [
        [[h0_idx, h0_p], [fwd_idx, round(h_fwd, 4)]],
        [[l0_idx, l0_p], [fwd_idx, round(l_fwd, 4)]],
        [[last_idx, round(breakout, 4)], [last_idx, round(target, 4)]],
    ]

    all_pts = [[int(p["index"]), float(p["price"])] for p in rh] + \
              [[int(p["index"]), float(p["price"])] for p in rl]
    all_pts.sort(key=lambda x: x[0])
    pt_labels = [f"H{i+1}" if any(p["index"] == pt[0] and p["price"] == pt[1] for p in rh)
                 else f"L{i+1}" for i, pt in enumerate(all_pts)]

    result.update({
        "pattern_name":       "Falling Wedge",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "bullish",
        "points":             all_pts,
        "point_labels":       pt_labels,
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "measured_move"],
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#22c55e",
            "opacity": 0.10,
        },
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Rising Wedge, Falling Wedge] detections.

    Multi-scale scan: tries progressively larger minimum-move scales so both
    short-term and full-chart wedges are caught.  Span bonus rewards
    patterns that encompass the entire chart window.
    """
    if price_data is None or len(price_data) < 50:
        return [base_result("Rising Wedge"), base_result("Falling Wedge")]

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    total_bars = len(df)

    best_rising  = base_result("Rising Wedge")
    best_falling = base_result("Falling Wedge")

    for scale in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        mm = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        highs = swings["swing_highs"]
        lows  = swings["swing_lows"]

        for detect_fn, best_key in [
            (_detect_rising_wedge, 'rising'),
            (_detect_falling_wedge, 'falling'),
        ]:
            r = detect_fn(df, highs, lows)
            conf = float(r.get("confidence", 0.0))
            if conf > 0:
                all_idx = [p["index"] for p in highs + lows]
                if all_idx:
                    span = (max(all_idx) - min(all_idx)) / max(total_bars, 1)
                    conf += span * 15
                    r["confidence"] = round(min(conf, 99.0), 2)
                    r["probability"] = r["confidence"]
            best_ref = best_rising if best_key == 'rising' else best_falling
            if conf > float(best_ref.get("confidence", 0.0)):
                if best_key == 'rising':
                    best_rising = r
                else:
                    best_falling = r

    return [best_rising, best_falling]
