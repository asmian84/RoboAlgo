"""Ascending / Descending Scallop detector — continuation patterns.

Ascending scallop (bullish): In an uptrend, price pulls back in a
shallow U-curve then breaks above the prior high.

Descending scallop (bearish): In a downtrend, price bounces in an
inverted-U curve then breaks below the prior low.

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


_SWING_WINDOW = 5
_MIN_BARS = 50


def _empty(name: str) -> dict[str, Any]:
    r = base_result(name)
    r["direction"] = "bullish" if "Ascending" in name else "bearish"
    return r


def _detect_ascending(df: pd.DataFrame, swings: dict) -> dict[str, Any]:
    """Ascending Scallop: U-shaped pullback in an uptrend."""
    name = "Ascending Scallop"
    result = _empty(name)

    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if len(highs) < 2 or len(lows) < 2:
        return result

    # Need an uptrend: recent highs should be ascending
    recent_highs = highs[-4:] if len(highs) >= 4 else highs
    if len(recent_highs) >= 2 and recent_highs[-1]["price"] <= recent_highs[0]["price"]:
        return result

    # Find U-shaped pullback: a swing high, followed by 1+ swing lows, then breakout
    # Use the last 2 swing highs and lows between them
    left_high = highs[-2]
    right_high = highs[-1]

    # Lows between the two highs form the scallop trough
    trough_lows = [l for l in lows if left_high["index"] < l["index"] < right_high["index"]]
    if not trough_lows:
        return result

    trough = min(trough_lows, key=lambda p: p["price"])

    # Scallop depth: pullback should be moderate (3-30%)
    pullback_pct = (left_high["price"] - trough["price"]) / max(left_high["price"], 1e-9)
    if pullback_pct < 0.03 or pullback_pct > 0.30:
        return result

    # Right high should be near or above left high (continuation)
    if right_high["price"] < left_high["price"] * 0.97:
        return result

    breakout = float(max(left_high["price"], right_high["price"]))
    invalidation = float(trough["price"])
    target = float(breakout + (breakout - trough["price"]))

    structure_quality = float(np.clip(70 + pullback_pct * 100 - abs(
        left_high["price"] - right_high["price"]
    ) / max(left_high["price"], 1e-9) * 200, 0, 100))

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
        # Scallop arc: left_high -> trough -> right_high
        [[left_high["index"], left_high["price"]], [trough["index"], trough["price"]]],
        [[trough["index"], trough["price"]], [right_high["index"], right_high["price"]]],
        # Breakout level
        [[left_high["index"], breakout], [last_idx, breakout]],
        # Measured move
        [[last_idx, breakout], [last_idx, target]],
    ]
    overlay_line_roles = ["scallop_left", "scallop_right", "breakout", "measured_move"]

    result.update({
        "status": status,
        "direction": "bullish",
        "breakout_level": round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4),
        "probability": round(probability, 2),
        "confidence": round(probability, 2),
        "points": [
            [left_high["index"], left_high["price"]],
            [trough["index"], trough["price"]],
            [right_high["index"], right_high["price"]],
        ],
        "point_labels": ["START", "TROUGH", "BREAKOUT"],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": overlay_line_roles,
    })
    return result


def _detect_descending(df: pd.DataFrame, swings: dict) -> dict[str, Any]:
    """Descending Scallop: inverted-U bounce in a downtrend."""
    name = "Descending Scallop"
    result = _empty(name)

    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if len(highs) < 2 or len(lows) < 2:
        return result

    # Need a downtrend: recent lows should be descending
    recent_lows = lows[-4:] if len(lows) >= 4 else lows
    if len(recent_lows) >= 2 and recent_lows[-1]["price"] >= recent_lows[0]["price"]:
        return result

    # Use the last 2 swing lows and highs between them
    left_low = lows[-2]
    right_low = lows[-1]

    peak_highs = [h for h in highs if left_low["index"] < h["index"] < right_low["index"]]
    if not peak_highs:
        return result

    peak = max(peak_highs, key=lambda p: p["price"])

    bounce_pct = (peak["price"] - left_low["price"]) / max(left_low["price"], 1e-9)
    if bounce_pct < 0.03 or bounce_pct > 0.30:
        return result

    if right_low["price"] > left_low["price"] * 1.03:
        return result

    breakout = float(min(left_low["price"], right_low["price"]))
    invalidation = float(peak["price"])
    target = float(breakout - (peak["price"] - breakout))

    structure_quality = float(np.clip(70 + bounce_pct * 100 - abs(
        left_low["price"] - right_low["price"]
    ) / max(left_low["price"], 1e-9) * 200, 0, 100))

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
        [[left_low["index"], left_low["price"]], [peak["index"], peak["price"]]],
        [[peak["index"], peak["price"]], [right_low["index"], right_low["price"]]],
        [[left_low["index"], breakout], [last_idx, breakout]],
        [[last_idx, breakout], [last_idx, target]],
    ]
    overlay_line_roles = ["scallop_left", "scallop_right", "breakdown", "measured_move"]

    result.update({
        "status": status,
        "direction": "bearish",
        "breakout_level": round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4),
        "probability": round(probability, 2),
        "confidence": round(probability, 2),
        "points": [
            [left_low["index"], left_low["price"]],
            [peak["index"], peak["price"]],
            [right_low["index"], right_low["price"]],
        ],
        "point_labels": ["START", "PEAK", "BREAKDOWN"],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": overlay_line_roles,
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect Ascending and Descending Scallop patterns.

    Parameters
    ----------
    symbol : str
        Ticker symbol (logging only).
    price_data : pd.DataFrame
        OHLCV DataFrame sorted oldest-first.

    Returns
    -------
    list[dict]
        Two pattern dicts (ascending + descending), each following the
        RoboAlgo service contract.
    """
    asc = _empty("Ascending Scallop")
    desc = _empty("Descending Scallop")

    if price_data is None or len(price_data) < _MIN_BARS:
        return [asc, desc]

    try:
        df = price_data.sort_values("date").reset_index(drop=True).copy()
        adaptive_mm = compute_adaptive_minimum_move(df)
        swings = detect_swings(df, minimum_move=adaptive_mm)

        asc = _detect_ascending(df, swings)
        desc = _detect_descending(df, swings)
    except Exception:
        pass

    return [asc, desc]
