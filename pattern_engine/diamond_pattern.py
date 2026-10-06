"""Diamond Top / Diamond Bottom detector — reversal patterns.

Diamond Top (bearish): Expanding then contracting price range at the
end of an uptrend, forming a diamond shape.

Diamond Bottom (bullish): Expanding then contracting price range at
the end of a downtrend.

Exposes:
    detect(symbol: str, df: pd.DataFrame) -> list[dict]
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


_MIN_BARS = 60
_LOOKBACK = 120


def _empty(name: str, direction: str) -> dict[str, Any]:
    r = base_result(name)
    r["direction"] = direction
    return r


def _range_expansion_contraction(df_window: pd.DataFrame, split_ratio: float = 0.5) -> tuple[bool, float]:
    """Check if the first half has expanding range and second half has contracting range.

    Returns (is_diamond, quality_score).
    """
    n = len(df_window)
    if n < 10:
        return False, 0.0

    highs = df_window["high"].astype(float).values
    lows = df_window["low"].astype(float).values
    ranges = highs - lows

    mid = int(n * split_ratio)
    if mid < 3 or n - mid < 3:
        return False, 0.0

    first_half = ranges[:mid]
    second_half = ranges[mid:]

    # First half: ranges should generally increase
    # Second half: ranges should generally decrease
    first_trend = np.polyfit(np.arange(len(first_half)), first_half, 1)[0]
    second_trend = np.polyfit(np.arange(len(second_half)), second_half, 1)[0]

    if first_trend <= 0 or second_trend >= 0:
        return False, 0.0

    # Quality based on how strong the expansion/contraction is
    avg_range = float(np.mean(ranges)) or 1e-9
    quality = float(np.clip(
        50 + (first_trend / avg_range) * 200 - (second_trend / avg_range) * 200, 0, 100
    ))

    return True, quality


def _detect_diamond_top(df: pd.DataFrame, swings: dict) -> dict[str, Any]:
    """Diamond Top: bearish reversal at end of uptrend."""
    name = "Diamond Top"
    result = _empty(name, "bearish")

    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if len(highs) < 3 or len(lows) < 3:
        return result

    # Check for prior uptrend
    lookback = min(_LOOKBACK, len(df))
    window = df.tail(lookback).reset_index(drop=True)
    pre_trend_end = max(0, len(window) - 40)
    if pre_trend_end < 10:
        return result

    pre_close = window["close"].astype(float).iloc[:pre_trend_end]
    if float(pre_close.iloc[-1]) <= float(pre_close.iloc[0]):
        return result  # No uptrend

    # Check diamond shape in last ~40 bars
    diamond_window = window.iloc[pre_trend_end:]
    if len(diamond_window) < 10:
        return result

    is_diamond, quality = _range_expansion_contraction(diamond_window)
    if not is_diamond:
        return result

    # Find the diamond outline from swing points within the diamond zone
    global_offset = len(df) - lookback + pre_trend_end
    diamond_highs = [h for h in highs if h["index"] >= global_offset]
    diamond_lows = [l for l in lows if l["index"] >= global_offset]

    if len(diamond_highs) < 2 or len(diamond_lows) < 2:
        return result

    # Diamond vertices: leftmost point, top, rightmost point, bottom
    top = max(diamond_highs, key=lambda p: p["price"])
    bottom = min(diamond_lows, key=lambda p: p["price"])
    all_points = diamond_highs + diamond_lows
    all_points.sort(key=lambda p: p["index"])
    left_point = all_points[0]
    right_point = all_points[-1]

    breakout = float(bottom["price"])
    invalidation = float(top["price"])
    diamond_height = top["price"] - bottom["price"]
    target = float(breakout - diamond_height)

    structure_quality = float(np.clip(quality, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    last_idx = len(df) - 1

    overlay_lines = [
        # Diamond outline: left -> top -> right -> bottom -> left
        [[left_point["index"], left_point["price"]], [top["index"], top["price"]]],
        [[top["index"], top["price"]], [right_point["index"], right_point["price"]]],
        [[right_point["index"], right_point["price"]], [bottom["index"], bottom["price"]]],
        [[bottom["index"], bottom["price"]], [left_point["index"], left_point["price"]]],
        # Breakdown level
        [[bottom["index"], breakout], [last_idx, breakout]],
        # Measured move
        [[last_idx, breakout], [last_idx, target]],
    ]
    overlay_line_roles = [
        "diamond_ul", "diamond_ur", "diamond_lr", "diamond_ll",
        "breakdown", "measured_move",
    ]

    result.update({
        "status": status,
        "direction": "bearish",
        "breakout_level": round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4),
        "probability": round(probability, 2),
        "confidence": round(probability, 2),
        "points": [
            [left_point["index"], left_point["price"]],
            [top["index"], top["price"]],
            [right_point["index"], right_point["price"]],
            [bottom["index"], bottom["price"]],
        ],
        "point_labels": ["LEFT", "TOP", "RIGHT", "BOTTOM"],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": overlay_line_roles,
        "fill_zone": {
            "upper": 0,
            "lower": 3,
            "color": "#ef4444",
            "opacity": 0.08,
        },
    })
    return result


def _detect_diamond_bottom(df: pd.DataFrame, swings: dict) -> dict[str, Any]:
    """Diamond Bottom: bullish reversal at end of downtrend."""
    name = "Diamond Bottom"
    result = _empty(name, "bullish")

    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if len(highs) < 3 or len(lows) < 3:
        return result

    # Check for prior downtrend
    lookback = min(_LOOKBACK, len(df))
    window = df.tail(lookback).reset_index(drop=True)
    pre_trend_end = max(0, len(window) - 40)
    if pre_trend_end < 10:
        return result

    pre_close = window["close"].astype(float).iloc[:pre_trend_end]
    if float(pre_close.iloc[-1]) >= float(pre_close.iloc[0]):
        return result  # No downtrend

    # Check diamond shape in last ~40 bars
    diamond_window = window.iloc[pre_trend_end:]
    if len(diamond_window) < 10:
        return result

    is_diamond, quality = _range_expansion_contraction(diamond_window)
    if not is_diamond:
        return result

    global_offset = len(df) - lookback + pre_trend_end
    diamond_highs = [h for h in highs if h["index"] >= global_offset]
    diamond_lows = [l for l in lows if l["index"] >= global_offset]

    if len(diamond_highs) < 2 or len(diamond_lows) < 2:
        return result

    top = max(diamond_highs, key=lambda p: p["price"])
    bottom = min(diamond_lows, key=lambda p: p["price"])
    all_points = diamond_highs + diamond_lows
    all_points.sort(key=lambda p: p["index"])
    left_point = all_points[0]
    right_point = all_points[-1]

    breakout = float(top["price"])
    invalidation = float(bottom["price"])
    diamond_height = top["price"] - bottom["price"]
    target = float(breakout + diamond_height)

    structure_quality = float(np.clip(quality, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    last_idx = len(df) - 1

    overlay_lines = [
        [[left_point["index"], left_point["price"]], [top["index"], top["price"]]],
        [[top["index"], top["price"]], [right_point["index"], right_point["price"]]],
        [[right_point["index"], right_point["price"]], [bottom["index"], bottom["price"]]],
        [[bottom["index"], bottom["price"]], [left_point["index"], left_point["price"]]],
        [[top["index"], breakout], [last_idx, breakout]],
        [[last_idx, breakout], [last_idx, target]],
    ]
    overlay_line_roles = [
        "diamond_ul", "diamond_ur", "diamond_lr", "diamond_ll",
        "breakout", "measured_move",
    ]

    result.update({
        "status": status,
        "direction": "bullish",
        "breakout_level": round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4),
        "probability": round(probability, 2),
        "confidence": round(probability, 2),
        "points": [
            [left_point["index"], left_point["price"]],
            [top["index"], top["price"]],
            [right_point["index"], right_point["price"]],
            [bottom["index"], bottom["price"]],
        ],
        "point_labels": ["LEFT", "TOP", "RIGHT", "BOTTOM"],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": overlay_line_roles,
        "fill_zone": {
            "upper": 0,
            "lower": 3,
            "color": "#22c55e",
            "opacity": 0.08,
        },
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect Diamond Top and Diamond Bottom patterns.

    Parameters
    ----------
    symbol : str
        Ticker symbol (logging only).
    price_data : pd.DataFrame
        OHLCV DataFrame sorted oldest-first.

    Returns
    -------
    list[dict]
        Two pattern dicts following the RoboAlgo service contract.
    """
    dt = _empty("Diamond Top", "bearish")
    db = _empty("Diamond Bottom", "bullish")

    if price_data is None or len(price_data) < _MIN_BARS:
        return [dt, db]

    try:
        df = price_data.sort_values("date").reset_index(drop=True).copy()
        adaptive_mm = compute_adaptive_minimum_move(df)
        swings = detect_swings(df, minimum_move=adaptive_mm)

        dt = _detect_diamond_top(df, swings)
        db = _detect_diamond_bottom(df, swings)
    except Exception:
        pass

    return [dt, db]
