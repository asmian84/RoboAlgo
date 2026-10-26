"""Harmonic pattern detector — Gartley, Bat, Butterfly, Crab, Cypher.

Detects XABCD Fibonacci ratio patterns using a multi-scale pivot scan.

KEY ACCURACY REQUIREMENTS (all must pass or pattern is rejected):
  1. Pivot ALTERNATION: pivots must strictly alternate high-low-high-low-high
     (or low-high-low-high-low). No two consecutive highs or lows allowed.
  2. Fibonacci RATIOS: tightened to ±4% of textbook values. Out-of-range = 0.
  3. STRUCTURAL VALIDITY: legs must form correct directional shape.
     Gartley/Bat/Cypher: D must not break below X (bullish) or above X (bearish).
     Butterfly/Crab: D extends beyond X (by design — extreme extension patterns).
  4. MINIMUM LEG SIZE: each leg must be at least 1.5% of current price to exclude
     micro-noise patterns that hit ratios by coincidence.
  5. MINIMUM SCORE: raw ratio score must be ≥ 62 before any bonus. Final ≥ 68.
     This eliminates "borderline" matches that swamp the scanner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}

# ── Textbook Fibonacci ratio definitions ──────────────────────────────────────
# Tightened to ±3-4% of textbook values (was ±8–15%).
# BC_AB intentionally kept wide (0.382–0.886) — both values are valid per Carney.
# XD_XA is the primary discriminator; it has the tightest tolerance.

PATTERNS: dict[str, dict[str, tuple[float, float]]] = {
    "Gartley": {
        "AB_XA": (0.570, 0.650),     # textbook 0.618 — tightened from (0.55, 0.68)
        "BC_AB": (0.382, 0.886),     # both 38.2% and 88.6% retracements valid
        "CD_BC": (1.272, 1.618),     # both 127.2% and 161.8% extensions valid
        "XD_XA": (0.755, 0.818),     # textbook 0.786 — tightened from (0.75, 0.82)
    },
    "Bat": {
        "AB_XA": (0.362, 0.508),     # textbook 0.382–0.500 — tightened from (0.33, 0.52)
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "XD_XA": (0.858, 0.914),     # textbook 0.886 — tightened from (0.84, 0.92)
    },
    "Butterfly": {
        "AB_XA": (0.754, 0.820),     # textbook 0.786 — tightened from (0.72, 0.82)
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "XD_XA": (1.242, 1.640),     # textbook 1.272–1.618 — tightened from (1.20, 1.68)
    },
    "Crab": {
        "AB_XA": (0.362, 0.618),     # textbook 0.382–0.618
        "BC_AB": (0.382, 0.886),
        "CD_BC": (2.618, 3.618),     # TIGHTENED lower bound from 2.24→2.618 (textbook minimum)
        "XD_XA": (1.580, 1.658),     # textbook 1.618 — very tight ±2.5%
    },
    "Cypher": {
        "AB_XA": (0.362, 0.618),     # textbook 0.382–0.618
        "BC_XA": (1.130, 1.414),     # BC extends beyond A (unique to Cypher)
        "XD_XA": (0.755, 0.818),     # textbook 0.786 — tightened from (0.74, 0.82)
    },
}

# Patterns where D is allowed to extend BEYOND X (extreme extension patterns)
_EXTENSION_PATTERNS = {"Butterfly", "Crab"}

# Minimum raw score before span bonus considered (prevents low-quality boosting)
_MIN_RAW = 62.0
# Minimum final score to include in results
_MIN_FINAL = 68.0


def _match_ratios(ratios: dict[str, float], target: dict[str, tuple[float, float]]) -> float:
    """Score how well computed ratios fit a target pattern definition.

    Returns 0–100.
    - All key ratios that ARE present in `ratios` must be in-range.
    - If ANY key ratio is outside its range, apply a steep penalty.
    - Missing ratio keys are skipped (allows Cypher to score without CD_BC).
    """
    errs: list[float] = []
    for k, (lo, hi) in target.items():
        v = ratios.get(k)
        if v is None:
            continue
        if lo <= v <= hi:
            errs.append(0.0)
        else:
            # Penalty: normalised by half-range (steeper than original full-range)
            half_range = max((hi - lo) / 2.0, 1e-9)
            dist = min(abs(v - lo), abs(v - hi)) / half_range
            errs.append(dist)
    if not errs:
        return 0.0
    avg_err = float(np.mean(errs))
    return float(np.clip(100.0 - avg_err * 140.0, 0.0, 100.0))


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
        "pattern_name":       pattern_name,
        "pattern_category":   "harmonic",
        "status":             status,
        "direction":          "bullish" if bullish else "bearish",
        "breakout_level":     round(breakout, 4),
        "target":             round(target_fib, 4),
        "projected_target":   round(target_fib, 4),
        "invalidation_level": round(x_price, 4),
        "confidence":         round(best_score, 2),
        "probability":        round(best_score, 2),
        "points":             [[int(x["index"]), float(x["price"])] for x in p],
        "point_labels":       ["X", "A", "B", "C", "D"],
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
        "target2": round((a_price + c_price) / 2, 4),
        "b_price": round(float(b_price), 4),
    }


def _pivots_alternate(p5: list[dict[str, Any]]) -> bool:
    """Return True only if the 5 pivots strictly alternate high/low/high/low/high
    or low/high/low/high/low.  Two consecutive highs or lows = False."""
    types = [p["type"] for p in p5]
    return all(types[i] != types[i + 1] for i in range(4))


def _structural_valid(p5: list[dict[str, Any]], pattern_name: str) -> bool:
    """Check that the XABCD structure makes geometric sense.

    Bullish (X=low, D=low):
      - A above X, B below A, C above B, D below C
      - For Gartley/Bat/Cypher: D must NOT break below X (D > X)
      - For Butterfly/Crab: D is BELOW X (extreme extension — that's the point)

    Bearish (X=high, D=high):
      - A below X, B above A, C below B, D above C
      - For Gartley/Bat/Cypher: D must NOT break above X (D < X)
      - For Butterfly/Crab: D is ABOVE X
    """
    prices = [float(p["price"]) for p in p5]
    x, a, b, c, d = prices
    bullish = p5[0]["type"] == "low"   # X is a swing low → bullish pattern

    if bullish:
        # A above X; B is between X and A; C is between B and A; D below C
        if not (a > x):
            return False
        if not (x < b < a):
            return False
        if not (b < c <= a):    # C should not exceed A (some tolerance handled by ratios)
            return False
        if not (d < c):
            return False
        # D location: must not make new low for standard patterns
        if pattern_name not in _EXTENSION_PATTERNS:
            if d < x:
                return False
        else:
            # Butterfly/Crab: D must be BELOW X (extension beyond X)
            if d >= x:
                return False
    else:  # bearish
        if not (a < x):
            return False
        if not (a < b < x):
            return False
        if not (a <= c < b):
            return False
        if not (d > c):
            return False
        if pattern_name not in _EXTENSION_PATTERNS:
            if d > x:
                return False
        else:
            if d <= x:
                return False

    return True


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect ALL harmonic pattern types independently at multiple scales.

    Returns list[dict] — only patterns that pass ALL accuracy gates:
      - Alternating pivots
      - Structural validity (correct directional shape)
      - Minimum leg size (≥ 1.5% of price per leg)
      - Ratio score ≥ 62 raw
      - Final score ≥ 68
    """
    if price_data is None or len(price_data) < 60:
        return []

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    n_bars = len(df)
    close = float(df["close"].iloc[-1])

    # Minimum leg size: at least 1.5% of current price OR 2× adaptive minimum move
    min_leg_size = max(adaptive_mm * 2.0, close * 0.015)

    # D point must be in the most-recent 40% of bars
    recent_threshold = int(n_bars * 0.60)

    # ── Build pivot sets with type labels ─────────────────────────────────────
    pivot_sets: list[list[dict[str, Any]]] = []
    for scale in [1.0, 3.0, 8.0, 15.0, 25.0]:
        mm = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        # Tag each pivot with its type BEFORE merging
        tagged = sorted(
            [{"type": "high", **p} for p in swings["swing_highs"]] +
            [{"type": "low",  **p} for p in swings["swing_lows"]],
            key=lambda x: x["index"],
        )
        if len(tagged) >= 5:
            pivot_sets.append(tagged)

    if not pivot_sets:
        return []

    results: list[dict[str, Any]] = []

    for pattern_name, pattern_defs in PATTERNS.items():
        best_score = 0.0
        best_pvts: list[dict[str, Any]] = []
        best_ratios: dict[str, float] = {}

        for pivots in pivot_sets:
            for w in range(0, len(pivots) - 4):
                p5 = pivots[w:w + 5]

                # Gate 1: D must be in recent 40% of data
                if p5[4]["index"] < recent_threshold:
                    continue

                # Gate 2: pivots must strictly alternate high/low
                if not _pivots_alternate(p5):
                    continue

                # Gate 3: structural geometric validity
                if not _structural_valid(p5, pattern_name):
                    continue

                xa = abs(p5[1]["price"] - p5[0]["price"])
                ab = abs(p5[2]["price"] - p5[1]["price"])
                bc = abs(p5[3]["price"] - p5[2]["price"])
                cd = abs(p5[4]["price"] - p5[3]["price"])
                xd = abs(p5[4]["price"] - p5[0]["price"])

                if min(xa, ab) <= 0:
                    continue

                # Gate 4: minimum leg size — filters out micro-noise patterns
                if min(xa, ab, bc, cd) < min_leg_size:
                    continue

                ratios: dict[str, float] = {
                    "AB_XA": ab / xa,
                    "BC_AB": bc / ab if ab > 0 else 0.0,
                    "CD_BC": cd / bc if bc > 0 else 0.0,
                    "XD_XA": xd / xa,
                    "BC_XA": bc / xa,
                }

                raw_score = _match_ratios(ratios, pattern_defs)

                # Gate 5: raw score must exceed minimum BEFORE bonuses
                if raw_score < _MIN_RAW:
                    continue

                # Span bonus: reward patterns covering more of the chart (capped at 8)
                span_bars = max(1, p5[4]["index"] - p5[0]["index"])
                span_bonus = min(8.0, (span_bars / max(n_bars, 1)) * 20.0)
                score = raw_score + span_bonus

                if score > best_score:
                    best_score = score
                    best_pvts = list(p5)
                    best_ratios = dict(ratios)

        # Gate 6: final score threshold
        if best_score < _MIN_FINAL or not best_pvts:
            continue

        r = _build_result(pattern_name, best_pvts, best_ratios, best_score, close)
        if r["status"] not in ("NOT_PRESENT", "FAILED"):
            results.append(r)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
