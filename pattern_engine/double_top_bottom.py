"""Double Top, Double Bottom, Triple Top, Triple Bottom detector.

Double Top (bearish reversal):
  - Two roughly equal swing highs separated by a trough (neckline)
  - Breakdown below neckline → target = neckline − (high − neckline)

Double Bottom (bullish reversal):
  - Two roughly equal swing lows separated by a peak (neckline)
  - Breakout above neckline → target = neckline + (neckline − low)

Triple Top / Triple Bottom: same logic with 3 peaks/troughs.
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

# Max % price difference between peaks/troughs to count as "equal"
_EQ = 0.05   # 5 %
_EQ3 = 0.075  # slightly looser for triple formations


# ── Double Top ─────────────────────────────────────────────────────────────

def _detect_double_top(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    result = base_result("Double Top")
    if len(highs) < 2 or not lows:
        return result

    last_idx = len(df) - 1
    best: tuple | None = None

    candidates = highs[-8:]
    for i in range(len(candidates) - 1):
        h1, h2 = candidates[i], candidates[i + 1]
        # Find deepest trough between the two highs
        t_cands = [l for l in lows if h1["index"] < l["index"] < h2["index"]]
        if not t_cands:
            continue
        trough = min(t_cands, key=lambda x: x["price"])
        diff = abs(h1["price"] - h2["price"]) / max(h1["price"], 1e-9)
        if diff > _EQ:
            continue
        peak_avg = (h1["price"] + h2["price"]) / 2.0
        neck = float(trough["price"])
        depth = peak_avg - neck
        if depth <= 0:
            continue
        if best is None or diff < best[0]:
            best = (diff, h1, h2, trough, peak_avg, neck, depth)

    if best is None:
        return result

    diff, h1, h2, trough, peak_avg, neck, depth = best
    breakout = neck
    target = neck - depth
    invalidation = peak_avg * 1.005

    sq = float(np.clip(80 - diff * 800, 0, 100))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    overlay_lines = [
        [[int(h1["index"]), float(h1["price"])],       [int(trough["index"]), float(trough["price"])]],
        [[int(trough["index"]), float(trough["price"])],[int(h2["index"]), float(h2["price"])]],
        [[int(h2["index"]), float(h2["price"])],        [last_idx, neck]],
        [[int(h1["index"]), neck],                      [last_idx, neck]],
    ]
    result.update({
        "pattern_name": "Double Top", "status": status,
        "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4), "confidence": round(prob, 2),
        "probability": round(prob, 2), "direction": "bearish",
        "points": [[int(h1["index"]), float(h1["price"])],
                   [int(h2["index"]), float(h2["price"])],
                   [int(trough["index"]), float(trough["price"])]],
        "point_labels": ["P1", "P2", ""],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": ["peak_down", "peak_up", "peak_down", "neckline"],
    })
    return result


# ── Double Bottom ──────────────────────────────────────────────────────────

def _detect_double_bottom(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    result = base_result("Double Bottom")
    if len(lows) < 2 or not highs:
        return result

    last_idx = len(df) - 1
    best: tuple | None = None

    candidates = lows[-8:]
    for i in range(len(candidates) - 1):
        l1, l2 = candidates[i], candidates[i + 1]
        p_cands = [h for h in highs if l1["index"] < h["index"] < l2["index"]]
        if not p_cands:
            continue
        peak = max(p_cands, key=lambda x: x["price"])
        diff = abs(l1["price"] - l2["price"]) / max(abs(l1["price"]), 1e-9)
        if diff > _EQ:
            continue
        bottom_avg = (l1["price"] + l2["price"]) / 2.0
        neck = float(peak["price"])
        rise = neck - bottom_avg
        if rise <= 0:
            continue
        if best is None or diff < best[0]:
            best = (diff, l1, l2, peak, bottom_avg, neck, rise)

    if best is None:
        return result

    diff, l1, l2, peak, bottom_avg, neck, rise = best
    breakout = neck
    target = neck + rise
    invalidation = bottom_avg * 0.995

    sq = float(np.clip(80 - diff * 800, 0, 100))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    overlay_lines = [
        [[int(l1["index"]), float(l1["price"])],   [int(peak["index"]), float(peak["price"])]],
        [[int(peak["index"]), float(peak["price"])],[int(l2["index"]), float(l2["price"])]],
        [[int(l2["index"]), float(l2["price"])],   [last_idx, neck]],
        [[int(l1["index"]), neck],                  [last_idx, neck]],
    ]
    result.update({
        "pattern_name": "Double Bottom", "status": status,
        "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4), "confidence": round(prob, 2),
        "probability": round(prob, 2), "direction": "bullish",
        "points": [[int(l1["index"]), float(l1["price"])],
                   [int(l2["index"]), float(l2["price"])],
                   [int(peak["index"]), float(peak["price"])]],
        "point_labels": ["B1", "B2", ""],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": ["bottom_up", "bottom_down", "bottom_up", "neckline"],
    })
    return result


# ── Triple Top ─────────────────────────────────────────────────────────────

def _detect_triple_top(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    result = base_result("Triple Top")
    if len(highs) < 3 or len(lows) < 2:
        return result

    last_idx = len(df) - 1
    best: tuple | None = None

    candidates = highs[-9:]
    for i in range(len(candidates) - 2):
        h1, h2, h3 = candidates[i], candidates[i + 1], candidates[i + 2]
        t1_c = [l for l in lows if h1["index"] < l["index"] < h2["index"]]
        t2_c = [l for l in lows if h2["index"] < l["index"] < h3["index"]]
        if not t1_c or not t2_c:
            continue
        t1 = min(t1_c, key=lambda x: x["price"])
        t2 = min(t2_c, key=lambda x: x["price"])
        prices = [h1["price"], h2["price"], h3["price"]]
        spread = (max(prices) - min(prices)) / max(prices)
        if spread > _EQ3:
            continue
        neck = min(t1["price"], t2["price"])
        peak_avg = sum(prices) / 3.0
        depth = peak_avg - neck
        if depth <= 0:
            continue
        if best is None or spread < best[0]:
            best = (spread, h1, h2, h3, t1, t2, peak_avg, neck, depth)

    if best is None:
        return result

    spread, h1, h2, h3, t1, t2, peak_avg, neck, depth = best
    breakout = neck
    target = neck - depth
    invalidation = peak_avg * 1.005

    sq = float(np.clip(82 - spread * 500, 0, 100))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    overlay_lines = [
        [[int(h1["index"]), float(h1["price"])], [int(t1["index"]), float(t1["price"])]],
        [[int(t1["index"]), float(t1["price"])], [int(h2["index"]), float(h2["price"])]],
        [[int(h2["index"]), float(h2["price"])], [int(t2["index"]), float(t2["price"])]],
        [[int(t2["index"]), float(t2["price"])], [int(h3["index"]), float(h3["price"])]],
        [[int(h3["index"]), float(h3["price"])], [last_idx, neck]],
        [[int(h1["index"]), neck],               [last_idx, neck]],
    ]
    result.update({
        "pattern_name": "Triple Top", "status": status,
        "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4), "confidence": round(prob, 2),
        "probability": round(prob, 2), "direction": "bearish",
        "points": [[int(h1["index"]), float(h1["price"])],
                   [int(h2["index"]), float(h2["price"])],
                   [int(h3["index"]), float(h3["price"])],
                   [int(t1["index"]), float(t1["price"])],
                   [int(t2["index"]), float(t2["price"])]],
        "point_labels": ["P1", "P2", "P3", "", ""],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": ["peak_down", "peak_up", "peak_down", "peak_up", "peak_down", "neckline"],
    })
    return result


# ── Triple Bottom ──────────────────────────────────────────────────────────

def _detect_triple_bottom(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    result = base_result("Triple Bottom")
    if len(lows) < 3 or len(highs) < 2:
        return result

    last_idx = len(df) - 1
    best: tuple | None = None

    candidates = lows[-9:]
    for i in range(len(candidates) - 2):
        l1, l2, l3 = candidates[i], candidates[i + 1], candidates[i + 2]
        p1_c = [h for h in highs if l1["index"] < h["index"] < l2["index"]]
        p2_c = [h for h in highs if l2["index"] < h["index"] < l3["index"]]
        if not p1_c or not p2_c:
            continue
        p1 = max(p1_c, key=lambda x: x["price"])
        p2 = max(p2_c, key=lambda x: x["price"])
        prices = [l1["price"], l2["price"], l3["price"]]
        spread = (max(prices) - min(prices)) / max(abs(min(prices)), 1e-9)
        if spread > _EQ3:
            continue
        neck = max(p1["price"], p2["price"])
        bottom_avg = sum(prices) / 3.0
        rise = neck - bottom_avg
        if rise <= 0:
            continue
        if best is None or spread < best[0]:
            best = (spread, l1, l2, l3, p1, p2, bottom_avg, neck, rise)

    if best is None:
        return result

    spread, l1, l2, l3, p1, p2, bottom_avg, neck, rise = best
    breakout = neck
    target = neck + rise
    invalidation = bottom_avg * 0.995

    sq = float(np.clip(82 - spread * 500, 0, 100))
    prob = composite_probability(sq, volume_confirmation(df),
                                 liquidity_alignment_score(df, breakout),
                                 market_regime_score(df), momentum_score(df))
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    overlay_lines = [
        [[int(l1["index"]), float(l1["price"])], [int(p1["index"]), float(p1["price"])]],
        [[int(p1["index"]), float(p1["price"])], [int(l2["index"]), float(l2["price"])]],
        [[int(l2["index"]), float(l2["price"])], [int(p2["index"]), float(p2["price"])]],
        [[int(p2["index"]), float(p2["price"])], [int(l3["index"]), float(l3["price"])]],
        [[int(l3["index"]), float(l3["price"])], [last_idx, neck]],
        [[int(l1["index"]), neck],               [last_idx, neck]],
    ]
    result.update({
        "pattern_name": "Triple Bottom", "status": status,
        "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
        "projected_target": round(target, 4), "confidence": round(prob, 2),
        "probability": round(prob, 2), "direction": "bullish",
        "points": [[int(l1["index"]), float(l1["price"])],
                   [int(l2["index"]), float(l2["price"])],
                   [int(l3["index"]), float(l3["price"])],
                   [int(p1["index"]), float(p1["price"])],
                   [int(p2["index"]), float(p2["price"])]],
        "point_labels": ["B1", "B2", "B3", "", ""],
        "overlay_lines": overlay_lines,
        "overlay_line_roles": ["bottom_up", "bottom_down", "bottom_up", "bottom_down", "bottom_up", "neckline"],
    })
    return result


# ── Entry point ────────────────────────────────────────────────────────────

def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Double Top, Double Bottom, Triple Top, Triple Bottom] detections."""
    _names = ("Double Top", "Double Bottom", "Triple Top", "Triple Bottom")
    if price_data is None or len(price_data) < 50:
        return [base_result(n) for n in _names]

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    return [
        _detect_double_top(df, highs, lows),
        _detect_double_bottom(df, highs, lows),
        _detect_triple_top(df, highs, lows),
        _detect_triple_bottom(df, highs, lows),
    ]
