"""Harmonic pattern detector — Gartley, Bat, Butterfly, Crab, Cypher.

Detects XABCD Fibonacci ratio patterns using a multi-scale pivot scan.
Each of the 5 pattern types is scanned independently so ALL currently-active
harmonics can be returned simultaneously, not just the single "best" one.

Scales: minor (×1), major (×3), macro (×8), structural (×15), cycle (×25)

Returns list[dict] — one entry per detected pattern type.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}

# ── Textbook Fibonacci ratio definitions ─────────────────────────────────────
# Each key maps to (low, high) inclusive tolerance.
# XD_XA is the key discriminator between patterns.

PATTERNS: dict[str, dict[str, tuple[float, float]]] = {
    "Gartley": {
        "AB_XA": (0.55, 0.68),       # textbook 0.618
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.272, 1.618),
        "XD_XA": (0.75, 0.82),       # textbook 0.786
    },
    "Bat": {
        "AB_XA": (0.33, 0.52),        # textbook 0.382-0.50
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "XD_XA": (0.84, 0.92),        # textbook 0.886
    },
    "Butterfly": {
        "AB_XA": (0.72, 0.82),        # textbook 0.786
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "XD_XA": (1.20, 1.68),        # textbook 1.27-1.618 (D beyond X)
    },
    "Crab": {
        "AB_XA": (0.33, 0.65),        # textbook 0.382-0.618
        "BC_AB": (0.382, 0.886),
        "CD_BC": (2.24, 3.618),
        "XD_XA": (1.55, 1.70),        # textbook 1.618
    },
    "Cypher": {
        "AB_XA": (0.33, 0.65),        # textbook 0.382-0.618
        "BC_XA": (1.13, 1.414),       # BC is EXTENSION of XA (unique to Cypher)
        "XD_XA": (0.74, 0.82),        # textbook 0.786
    },
}


def _match_ratios(ratios: dict[str, float], target: dict[str, tuple[float, float]]) -> float:
    """Score how well computed ratios fit a target pattern definition.

    Returns 0-100.  Missing ratio keys in `ratios` are skipped (not penalised),
    allowing Cypher (which lacks CD_BC) to score correctly.
    """
    errs: list[float] = []
    for k, (lo, hi) in target.items():
        v = ratios.get(k)
        if v is None:
            continue  # skip — this ratio isn't relevant for the pattern
        if lo <= v <= hi:
            errs.append(0.0)
        else:
            dist = min(abs(v - lo), abs(v - hi)) / max((hi - lo), 1e-9)
            errs.append(dist)
    if not errs:
        return 0.0
    avg_err = float(np.mean(errs))
    return float(np.clip(100 - avg_err * 120, 0, 100))


def _compute_status(
    bullish: bool, close: float,
    x_price: float, breakout: float, target_fib: float,
) -> str:
    if bullish and close < x_price:
        return "FAILED"
    if not bullish and close > x_price:
        return "FAILED"
    if bullish:
        if close >= target_fib:
            return "COMPLETED"
        if close > breakout:
            return "BREAKOUT"
        if close >= breakout * 0.995:
            return "READY"
        return "FORMING"
    else:
        if close <= target_fib:
            return "COMPLETED"
        if close < breakout:
            return "BREAKOUT"
        if close <= breakout * 1.005:
            return "READY"
        return "FORMING"


def _build_result(
    pattern_name: str,
    best_pvts: list[dict[str, Any]],
    best_ratios: dict[str, float],
    best_score: float,
    close: float,
) -> dict[str, Any]:
    """Construct the full result dict for a detected harmonic pattern."""
    p = best_pvts
    x_price = float(p[0]["price"])
    a_price = float(p[1]["price"])
    b_price = float(p[2]["price"])
    c_price = float(p[3]["price"])
    d_price = float(p[4]["price"])
    xa = abs(a_price - x_price)

    bullish = d_price < c_price
    breakout = float(max(c_price, d_price) if bullish else min(c_price, d_price))
    target_fib = float(breakout + xa * 0.618 if bullish else breakout - xa * 0.618)

    status = _compute_status(bullish, close, x_price, breakout, target_fib)

    # PRZ (Potential Reversal Zone): ±1.5% band around D point
    prz_low  = round(d_price * 0.985, 4)
    prz_high = round(d_price * 1.015, 4)

    zigzag_segs = [
        [[int(p[i]["index"]), float(p[i]["price"])],
         [int(p[i + 1]["index"]), float(p[i + 1]["price"])]]
        for i in range(4)
    ]
    ref_segs = [
        [[int(p[1]["index"]), float(a_price)], [int(p[3]["index"]), float(c_price)]],
        [[int(p[0]["index"]), float(x_price)], [int(p[4]["index"]), float(d_price)]],
    ]

    return {
        "pattern_name":     pattern_name,
        "pattern_category": "harmonic",
        "status":           status,
        "direction":        "bullish" if bullish else "bearish",
        "breakout_level":   round(breakout, 4),
        "target":           round(target_fib, 4),
        "projected_target": round(target_fib, 4),
        "invalidation_level": round(x_price, 4),
        "confidence":       round(best_score, 2),
        "probability":      round(best_score, 2),
        "points": [[int(x["index"]), float(x["price"])] for x in p],
        "point_labels": ["X", "A", "B", "C", "D"],
        "ratios": {
            "AB_XA": round(best_ratios.get("AB_XA", 0.0), 3),
            "BC_AB": round(best_ratios.get("BC_AB", 0.0), 3),
            "CD_BC": round(best_ratios.get("CD_BC", 0.0), 3),
            "XD_XA": round(best_ratios.get("XD_XA", 0.0), 3),
            "BC_XA": round(best_ratios.get("BC_XA", 0.0), 3),
        },
        "prz_low":  prz_low,
        "prz_high": prz_high,
        "overlay_lines":      zigzag_segs + ref_segs,
        "overlay_line_roles": ["xa", "ab", "bc", "cd", "harm_ref_ac", "harm_ref_xd"],
        # Unused by new code but kept for backward compat with older API consumers
        "target2": round((a_price + c_price) / 2, 4),
        "b_price": round(float(b_price), 4),
    }


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect ALL harmonic pattern types independently at multiple scales.

    Returns list[dict] — one entry per detected and non-failed pattern type
    (Gartley, Bat, Butterfly, Crab, Cypher).  Pattern types with no valid
    match are omitted entirely.

    The service layer handles list[dict] returns correctly, so this change is
    backward-compatible with the existing PatternService pipeline.
    """
    if price_data is None or len(price_data) < 60:
        return []

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    n_bars = len(df)

    # D point must be in the most-recent 40% of bars (active / forming)
    recent_threshold = int(n_bars * 0.60)
    close = float(df["close"].iloc[-1])

    # ── Build pivot sets once for all 5 scales ────────────────────────────
    pivot_sets: list[list[dict[str, Any]]] = []
    for scale in [1.0, 3.0, 8.0, 15.0, 25.0]:
        mm = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        pivots = sorted(
            swings["swing_highs"] + swings["swing_lows"],
            key=lambda x: x["index"],
        )
        if len(pivots) >= 5:
            pivot_sets.append(pivots)

    if not pivot_sets:
        return []

    results: list[dict[str, Any]] = []

    # ── For each pattern type, find the best independent match ────────────
    for pattern_name, pattern_defs in PATTERNS.items():
        best_score = 0.0
        best_pvts: list[dict[str, Any]] = []
        best_ratios: dict[str, float] = {}

        for pivots in pivot_sets:
            for w in range(0, len(pivots) - 4):
                p5 = pivots[w:w + 5]
                # D must be in the recent portion of data
                if p5[4]["index"] < recent_threshold:
                    continue

                xa = abs(p5[1]["price"] - p5[0]["price"])
                ab = abs(p5[2]["price"] - p5[1]["price"])
                bc = abs(p5[3]["price"] - p5[2]["price"])
                cd = abs(p5[4]["price"] - p5[3]["price"])
                xd = abs(p5[4]["price"] - p5[0]["price"])

                if min(xa, ab) <= 0:
                    continue

                ratios: dict[str, float] = {
                    "AB_XA": ab / xa,
                    "BC_AB": bc / ab if ab > 0 else 0.0,
                    "CD_BC": cd / bc if bc > 0 else 0.0,
                    "XD_XA": xd / xa,
                    "BC_XA": bc / xa,   # for Cypher
                }

                raw_score = _match_ratios(ratios, pattern_defs)
                if raw_score < 40:
                    continue

                # Span bonus: reward patterns covering more of the chart
                span_bars = max(1, p5[4]["index"] - p5[0]["index"])
                span_bonus = min(10.0, (span_bars / max(n_bars, 1)) * 25.0)
                score = raw_score + span_bonus

                if score > best_score:
                    best_score = score
                    best_pvts = list(p5)
                    best_ratios = dict(ratios)

        if best_score < 45 or not best_pvts:
            continue

        r = _build_result(pattern_name, best_pvts, best_ratios, best_score, close)
        if r["status"] not in ("NOT_PRESENT", "FAILED"):
            results.append(r)

    # Sort by confidence descending so the best pattern appears first in the
    # pattern list (useful for scanner display)
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
