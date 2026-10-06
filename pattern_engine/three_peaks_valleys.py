"""3 Rising Valleys / 3 Descending Peaks detector — continuation patterns.

3 Rising Valleys (bullish): Three consecutive higher lows with
resistance breakout.

3 Descending Peaks (bearish): Three consecutive lower highs with
support breakdown.

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


_MIN_BARS = 50


def _empty(name: str, direction: str) -> dict[str, Any]:
    r = base_result(name)
    r["direction"] = direction
    return r


def _detect_rising_valleys(df: pd.DataFrame, swings: dict) -> dict[str, Any]:
    """Three consecutive higher swing lows + breakout above resistance."""
    name = "3 Rising Valleys"
    result = _empty(name, "bullish")

    lows = swings["swing_lows"]
    highs = swings["swing_highs"]
    if len(lows) < 3 or len(highs) < 2:
        return result

    # Scan for best triple of rising lows among the recent swing lows
    best: dict[str, Any] | None = None
    best_score = -1.0

    candidates = lows[-6:] if len(lows) >= 6 else lows
    for i in range(len(candidates) - 2):
        v1, v2, v3 = candidates[i], candidates[i + 1], candidates[i + 2]
        # Must be strictly rising
        if not (v2["price"] > v1["price"] and v3["price"] > v2["price"]):
            continue

        # Quality: how evenly spaced are the rises?
        rise1 = v2["price"] - v1["price"]
        rise2 = v3["price"] - v2["price"]
        symmetry = 1.0 - abs(rise1 - rise2) / max(rise1, rise2, 1e-9)

        # Resistance line: highest high between v1 and v3
        resistance_highs = [h for h in highs if v1["index"] <= h["index"] <= v3["index"]]
        if not resistance_highs:
            continue
        resistance = max(resistance_highs, key=lambda p: p["price"])

        score = symmetry * 50 + (v3["price"] - v1["price"]) / max(v1["price"], 1e-9) * 200
        if score > best_score:
            best_score = score
            best = {
                "v1": v1, "v2": v2, "v3": v3,
                "resistance": resistance,
                "symmetry": symmetry,
            }

    if best is None:
        return result

    v1, v2, v3 = best["v1"], best["v2"], best["v3"]
    resistance = best["resistance"]

    breakout = float(resistance["price"])
    invalidation = float(v3["price"])
    target = float(breakout + (breakout - v1["price"]))

    structure_quality = float(np.clip(60 + best["symmetry"] * 30, 0, 100))
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
        # Connect the three valleys
        [[v1["index"], v1["price"]], [v2["index"], v2["price"]]],
        [[v2["index"], v2["price"]], [v3["index"], v3["price"]]],
        # Resistance line
        [[v1["index"], breakout], [last_idx, breakout]],
        # Measured move
        [[last_idx, breakout], [last_idx, target]],
    ]
    overlay_line_roles = ["valley_trend_1", "valley_trend_2", "resistance", "measured_move"]

    result.update({
        "status": status,
        "direction": "bullish",
        "breakout_level": round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4),
        "probability": round(probability, 2),
        "confidence": round(probability, 2),
        "points": [
            [v1["index"], v1["price"]],
            [v2["index"], v2["price"]],
            [v3["index"], v3["price"]],
            [resistance["index"], resistance["price"]],
        ],
        "point_labels": ["V1", "V2", "V3", "RESISTANCE"],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": overlay_line_roles,
    })
    return result


def _detect_descending_peaks(df: pd.DataFrame, swings: dict) -> dict[str, Any]:
    """Three consecutive lower swing highs + breakdown below support."""
    name = "3 Descending Peaks"
    result = _empty(name, "bearish")

    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if len(highs) < 3 or len(lows) < 2:
        return result

    best: dict[str, Any] | None = None
    best_score = -1.0

    candidates = highs[-6:] if len(highs) >= 6 else highs
    for i in range(len(candidates) - 2):
        p1, p2, p3 = candidates[i], candidates[i + 1], candidates[i + 2]
        if not (p2["price"] < p1["price"] and p3["price"] < p2["price"]):
            continue

        drop1 = p1["price"] - p2["price"]
        drop2 = p2["price"] - p3["price"]
        symmetry = 1.0 - abs(drop1 - drop2) / max(drop1, drop2, 1e-9)

        support_lows = [l for l in lows if p1["index"] <= l["index"] <= p3["index"]]
        if not support_lows:
            continue
        support = min(support_lows, key=lambda p: p["price"])

        score = symmetry * 50 + (p1["price"] - p3["price"]) / max(p1["price"], 1e-9) * 200
        if score > best_score:
            best_score = score
            best = {
                "p1": p1, "p2": p2, "p3": p3,
                "support": support,
                "symmetry": symmetry,
            }

    if best is None:
        return result

    p1, p2, p3 = best["p1"], best["p2"], best["p3"]
    support = best["support"]

    breakout = float(support["price"])
    invalidation = float(p3["price"])
    target = float(breakout - (p1["price"] - breakout))

    structure_quality = float(np.clip(60 + best["symmetry"] * 30, 0, 100))
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
        [[p1["index"], p1["price"]], [p2["index"], p2["price"]]],
        [[p2["index"], p2["price"]], [p3["index"], p3["price"]]],
        [[p1["index"], breakout], [last_idx, breakout]],
        [[last_idx, breakout], [last_idx, target]],
    ]
    overlay_line_roles = ["peak_trend_1", "peak_trend_2", "support", "measured_move"]

    result.update({
        "status": status,
        "direction": "bearish",
        "breakout_level": round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4),
        "probability": round(probability, 2),
        "confidence": round(probability, 2),
        "points": [
            [p1["index"], p1["price"]],
            [p2["index"], p2["price"]],
            [p3["index"], p3["price"]],
            [support["index"], support["price"]],
        ],
        "point_labels": ["P1", "P2", "P3", "SUPPORT"],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": overlay_line_roles,
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect 3 Rising Valleys and 3 Descending Peaks patterns.

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
    rv = _empty("3 Rising Valleys", "bullish")
    dp = _empty("3 Descending Peaks", "bearish")

    if price_data is None or len(price_data) < _MIN_BARS:
        return [rv, dp]

    try:
        df = price_data.sort_values("date").reset_index(drop=True).copy()
        adaptive_mm = compute_adaptive_minimum_move(df)
        swings = detect_swings(df, minimum_move=adaptive_mm)

        rv = _detect_rising_valleys(df, swings)
        dp = _detect_descending_peaks(df, swings)
    except Exception:
        pass

    return [rv, dp]
