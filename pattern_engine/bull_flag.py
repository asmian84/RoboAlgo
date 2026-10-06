"""Bull Flag detector using swing highs/lows."""

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
    """Multi-scale bull flag detection.

    Tries multiple pole window sizes (35, 80, 150) to catch both
    short-term and long-term flags.  Larger flags get a span bonus.
    """
    if price_data is None or len(price_data) < 60:
        return base_result("Bull Flag")

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    total_bars = len(df)

    best = base_result("Bull Flag")
    best_conf = 0.0

    # Multi-scale: try progressively larger pole windows
    for pole_size, max_pole_bars in [(35, 16), (80, 40), (150, 80)]:
        if total_bars < pole_size + 25:
            continue
        r = _detect_bull_flag_at_scale(df, adaptive_mm, pole_size, max_pole_bars, total_bars)
        conf = float(r.get("confidence", 0.0))
        if conf > best_conf:
            best_conf = conf
            best = r

    return best


def _detect_bull_flag_at_scale(
    df: pd.DataFrame, adaptive_mm: float, pole_size: int, max_pole_bars: int, total_bars: int,
) -> dict[str, Any]:
    result = base_result("Bull Flag")
    close = df["close"].astype(float)
    pole_window = close.tail(pole_size).reset_index(drop=True)
    low_idx = int(pole_window.idxmin())
    high_idx = int(pole_window.idxmax())
    if high_idx <= low_idx:
        return result

    pole_move = (float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])) / max(float(pole_window.iloc[low_idx]), 1e-9)
    # ATR-adaptive pole minimum: large-caps have smaller % moves
    min_pole = max(3.0 * adaptive_mm, 0.03)
    if pole_move < min_pole or high_idx - low_idx > max_pole_bars:
        return result

    global_start = len(close) - len(pole_window) + high_idx
    post = df.iloc[global_start:].reset_index()
    swings = detect_swings(post.rename(columns={"index": "orig_index"}), minimum_move=adaptive_mm)
    highs = swings["swing_highs"][-4:]
    lows = swings["swing_lows"][-4:]
    if len(highs) < 2 or len(lows) < 2:
        return result

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows], dtype=float)
    yl = np.array([p["price"] for p in lows], dtype=float)
    h_slope, h_int = np.polyfit(xh, yh, 1)
    l_slope, l_int = np.polyfit(xl, yl, 1)

    if not (h_slope < 0 and l_slope < 0):
        return result

    last_idx = len(df) - 1
    local_last = int(last_idx - post["index"].iloc[0])
    breakout = float(h_slope * local_last + h_int)
    invalidation = float(l_slope * local_last + l_int)
    target = float(breakout + (float(pole_window.iloc[high_idx]) - float(pole_window.iloc[low_idx])))

    structure_quality = float(np.clip(65 + pole_move * 150 - abs(h_slope - l_slope) * 30, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    # Span bonus: larger flags are more significant
    span = (last_idx - (len(close) - len(pole_window) + low_idx)) / max(total_bars, 1)
    probability = min(probability + span * 15, 99.0)
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    first_idx = int(min(xh[0], xl[0]))
    upper_now = float(h_slope * local_last + h_int)
    lower_now = float(l_slope * local_last + l_int)

    # Convert local (post sub-DF, 0-based) indices → global bar indices so
    # service.py's _idx_to_date maps to the correct dates in the full history.
    g_first = global_start + first_idx
    # global_start + local_last == global_start + (last_idx - global_start) == last_idx
    # Pole: from pole bottom → pole top (global bar indices)
    pole_bottom_global = len(close) - len(pole_window) + low_idx
    pole_top_global    = global_start   # == len(close) - len(pole_window) + high_idx
    pole_bottom_price  = float(pole_window.iloc[low_idx])
    pole_top_price     = float(pole_window.iloc[high_idx])

    # ── Draw flag channel through ACTUAL swing pivots, extended forward ──
    fwd_idx = last_idx + 10
    # Upper: first flag high → last flag high → extended
    fh0 = highs[0]; fh1 = highs[-1]
    gh0, gh1 = global_start + int(fh0["index"]), global_start + int(fh1["index"])
    h_run = max(gh1 - gh0, 1)
    h_sl = (float(fh1["price"]) - float(fh0["price"])) / h_run
    h_fwd = float(fh1["price"]) + h_sl * (fwd_idx - gh1)
    # Lower: first flag low → last flag low → extended
    fl0 = lows[0]; fl1 = lows[-1]
    gl0, gl1 = global_start + int(fl0["index"]), global_start + int(fl1["index"])
    l_run = max(gl1 - gl0, 1)
    l_sl = (float(fl1["price"]) - float(fl0["price"])) / l_run
    l_fwd = float(fl1["price"]) + l_sl * (fwd_idx - gl1)

    overlay_lines = [
        # Flagpole: thick bold line
        [[pole_bottom_global, pole_bottom_price], [pole_top_global, pole_top_price]],
        # Flag channel: through actual swing highs, extended forward
        [[gh0, float(fh0["price"])], [fwd_idx, round(h_fwd, 4)]],
        # Flag channel: through actual swing lows, extended forward
        [[gl0, float(fl0["price"])], [fwd_idx, round(l_fwd, 4)]],
        # Measured move: breakout → target (pole projection)
        [[last_idx, round(breakout, 4)], [last_idx, round(target, 4)]],
    ]

    result.update(
        {
            "status": status,
            "breakout_level": round(breakout, 4),
            "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4),
            "probability": round(probability, 2),
            "confidence": round(probability, 2),
            "direction": "bullish",
            "points": [[global_start + p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
            "point_labels": ["FLAG H1", "FLAG H2", "FLAG L1", "FLAG L2"],
            "overlay_lines": overlay_lines,
            "overlay_line_roles": ["pole", "resistance", "support", "measured_move"],
            "fill_zone": {"upper": 1, "lower": 2, "color": "#00C805", "opacity": 0.15},
        }
    )
    return result

