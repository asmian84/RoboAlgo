"""Gann level detector — 9 fan angles, retracement levels, time cycles,
Square-of-9 price levels, and optional bearish fan from swing high.

Uses a swing-anchor approach: finds the most recent significant swing low,
projects 9 Gann fan lines (8x1 through 1x8) forward, computes price adherence
to the 1x1 line, and identifies retracement levels (1/8 through 7/8) plus
time cycle projections.

Also computes:
  - Square-of-9 price levels: harmonic price resistance/support derived from
    the current price rotated around the Gann square (each 90° rotation = +0.25
    units in sqrt-price space).
  - Bearish Gann fan from the most recent significant swing HIGH when price is
    below it (downtrend confirmation / resistance fan lines).

Confidence is based on actual price-fan adherence (% of bars near a fan line
+ trend consistency relative to the 1x1), NOT the tautological slope ratio
from the old version.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}

# ── Gann Fan Angle Definitions ────────────────────────────────────────────────
# Each angle is (name, price_units_per_time_unit_multiplier)
# The 1x1 angle has a multiplier of 1.0 (45° in Gann's square)
# Steeper angles (2x1, 3x1, …) move 2, 3, … price units per time unit
# Shallower angles (1x2, 1x3, …) move 0.5, 0.333, … price units per time unit

GANN_ANGLES: list[tuple[str, float]] = [
    ("8x1", 8.0),
    ("4x1", 4.0),
    ("3x1", 3.0),
    ("2x1", 2.0),
    ("1x1", 1.0),   # the 45° master angle
    ("1x2", 0.5),
    ("1x3", 1.0 / 3.0),
    ("1x4", 0.25),
    ("1x8", 0.125),
]

# Gann retracement levels (divisions of the price range)
GANN_RETRACEMENTS = [
    (0.125, "1/8"),
    (0.250, "2/8 (25%)"),
    (0.333, "1/3"),
    (0.375, "3/8"),
    (0.500, "4/8 (50%)"),
    (0.625, "5/8"),
    (0.667, "2/3"),
    (0.750, "6/8 (75%)"),
    (0.875, "7/8"),
]

# Gann time cycle projections (bars from anchor)
GANN_TIME_CYCLES = [30, 60, 90, 120, 180, 360]


def _square_of_9_levels(anchor_price: float, current_price: float) -> list[dict[str, Any]]:
    """Compute Square-of-9 harmonic price levels around the current price.

    Algorithm:
      n = sqrt(current_price)
      Each 90° rotation in the Gann square = +0.25 in sqrt-space.
      Levels are (n + k * 0.25)² for k in range that stays within ±40% of price.
    Returns a sorted list of {price, label, rotation} dicts.
    """
    if current_price <= 0:
        return []
    n = math.sqrt(current_price)
    levels: list[dict[str, Any]] = []
    for k in range(-12, 13):          # ±3 full rotations (±360°)
        val = (n + k * 0.25) ** 2
        if val <= 0:
            continue
        pct_from_current = abs(val - current_price) / current_price
        if pct_from_current > 0.40:   # only show levels within ±40% of price
            continue
        rotation = ((k % 4) + 4) % 4 * 90   # 0, 90, 180, or 270 degrees
        levels.append({
            "price":    round(val, 4),
            "label":    f"SQ9 {rotation}°",
            "rotation": rotation,
            "k":        k,
        })
    # Remove near-duplicates (within 0.3% of each other)
    sorted_lvls = sorted(levels, key=lambda x: x["price"])
    deduped: list[dict[str, Any]] = []
    for lvl in sorted_lvls:
        if not deduped or abs(lvl["price"] - deduped[-1]["price"]) / max(deduped[-1]["price"], 1) > 0.003:
            deduped.append(lvl)
    return deduped


def _price_fan_adherence(
    close: pd.Series,
    anchor_idx: int,
    anchor_price: float,
    slope_1x1: float,
    tolerance_pct: float = 0.02,
) -> float:
    """Measure what % of bars are within `tolerance_pct` of ANY fan line.

    This replaces the old tautological confidence that always returned 85.
    """
    if slope_1x1 <= 0 or len(close) == 0:
        return 0.0

    hits = 0
    total = 0
    for i in range(len(close)):
        if i <= anchor_idx:
            continue
        dt = i - anchor_idx
        px = float(close.iloc[i])
        total += 1

        # Check proximity to each fan line
        for _, mult in GANN_ANGLES:
            fan_price = anchor_price + slope_1x1 * mult * dt
            if fan_price <= 0:
                continue
            if abs(px - fan_price) / fan_price <= tolerance_pct:
                hits += 1
                break  # one hit per bar is enough

    return hits / max(total, 1)


def _trend_consistency(
    close: pd.Series,
    anchor_idx: int,
    anchor_price: float,
    slope_1x1: float,
) -> float:
    """Score 0-1: how consistently price stays above (bullish) or below (bearish)
    the 1x1 line.  High consistency = strong Gann structure.
    """
    if slope_1x1 <= 0 or len(close) == 0:
        return 0.0

    above = 0
    total = 0
    for i in range(len(close)):
        if i <= anchor_idx:
            continue
        dt = i - anchor_idx
        gann_1x1 = anchor_price + slope_1x1 * dt
        if float(close.iloc[i]) >= gann_1x1:
            above += 1
        total += 1

    if total == 0:
        return 0.0
    ratio = above / total
    # Score is highest when price is consistently on one side (>0.8 or <0.2)
    return max(abs(ratio - 0.5) * 2.0, 0.0)


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    """Detect Gann levels with fan lines, retracements, and time cycles."""
    result = {
        "pattern_name": "Gann Levels",
        "pattern_category": "gann",
        "status": "NOT_PRESENT",
        "breakout_level": None,
        "target": None,
        "invalidation_level": None,
        "confidence": 0.0,
        "points": [],
        "overlay_lines": [],
        "fan_lines": [],
        "retracement_levels": [],
        "time_cycles": [],
    }
    if price_data is None or len(price_data) < 60:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    lows = swings["swing_lows"]
    highs = swings["swing_highs"]
    if not lows or not highs:
        return result

    # ── Find anchor: scan backward through swing lows for one with a future high ──
    anchor_low = None
    pivot_high = None
    for candidate in reversed(lows[-10:]):  # check last 10 swing lows
        future_highs = [h for h in highs if h["index"] > candidate["index"]]
        if future_highs:
            anchor_low = candidate
            pivot_high = max(future_highs, key=lambda h: h["price"])
            if pivot_high["price"] > anchor_low["price"]:
                break  # found a valid anchor with a higher subsequent high

    if anchor_low is None or pivot_high is None:
        return result

    bars = max(pivot_high["index"] - anchor_low["index"], 1)
    rise = pivot_high["price"] - anchor_low["price"]
    if rise <= 0:
        return result

    # ── Compute the 1x1 slope (price per bar at 45°) ─────────────────────
    slope_1x1 = rise / bars

    last_idx = len(df) - 1
    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    # ── Project all 9 fan lines to current bar ────────────────────────────
    dt = last_idx - anchor_low["index"]
    fan_lines: list[dict[str, Any]] = []
    overlay_lines: list[list[list[int | float]]] = []

    for name, mult in GANN_ANGLES:
        fan_price = anchor_low["price"] + slope_1x1 * mult * dt
        fan_lines.append({
            "angle": name,
            "multiplier": mult,
            "current_price": round(float(fan_price), 4),
        })
        # Overlay line from anchor to current projection
        overlay_lines.append([
            [int(anchor_low["index"]), float(anchor_low["price"])],
            [int(last_idx), round(float(fan_price), 4)],
        ])

    # ── Key levels from fan projections ───────────────────────────────────
    gann_1x1 = anchor_low["price"] + slope_1x1 * dt
    gann_2x1 = anchor_low["price"] + slope_1x1 * 2.0 * dt
    gann_1x2 = anchor_low["price"] + slope_1x1 * 0.5 * dt

    breakout = float(gann_1x1)
    target = float(gann_2x1)
    invalidation = float(gann_1x2)

    # ── Gann Retracement Levels ───────────────────────────────────────────
    range_low = float(anchor_low["price"])
    range_high = float(pivot_high["price"])
    retracement_levels: list[dict[str, Any]] = []
    for frac, label in GANN_RETRACEMENTS:
        level = range_high - (range_high - range_low) * frac
        retracement_levels.append({
            "fraction": frac,
            "label": label,
            "price": round(level, 4),
        })

    # ── Time Cycle Projections ────────────────────────────────────────────
    time_cycles: list[dict[str, Any]] = []
    for cycle_bars in GANN_TIME_CYCLES:
        projected_idx = anchor_low["index"] + cycle_bars
        if projected_idx <= last_idx + 60:  # only show cycles within 60 bars of current
            time_cycles.append({
                "cycle_bars": cycle_bars,
                "anchor_index": int(anchor_low["index"]),
                "projected_index": int(projected_idx),
                "status": "past" if projected_idx <= last_idx else "future",
            })

    # ── Square-of-9 harmonic price levels ─────────────────────────────────
    square_of_9 = _square_of_9_levels(anchor_low["price"], last_close)

    # ── Bearish fan from most recent swing HIGH (downtrend pressure lines) ─
    # Only project if price is currently below the highest recent swing high,
    # meaning a bearish fan is structurally active.
    bearish_fan_lines: list[dict[str, Any]] = []
    bearish_overlay_lines: list[list[list[int | float]]] = []
    if highs:
        # Use the highest swing high that is at least 5% above the anchor low
        valid_highs = [h for h in highs if h["price"] > anchor_low["price"] * 1.05]
        if valid_highs:
            anchor_high = max(valid_highs, key=lambda h: h["price"])
            # Bearish 1x1 slope: price falls by equal units per bar from the peak
            bear_bars   = max(last_idx - anchor_high["index"], 1)
            bear_range  = anchor_high["price"] - float(close.min())
            bear_1x1    = bear_range / bear_bars          # price drop per bar at 1x1
            bear_dt     = last_idx - anchor_high["index"]
            for name, mult in GANN_ANGLES:
                # Bearish fan: lines project downward (subtract slope)
                fan_price_at_end = anchor_high["price"] - bear_1x1 * mult * bear_dt
                bearish_fan_lines.append({
                    "angle": name,
                    "multiplier": mult,
                    "current_price": round(float(fan_price_at_end), 4),
                })
                bearish_overlay_lines.append([
                    [int(anchor_high["index"]), float(anchor_high["price"])],
                    [int(last_idx), round(float(fan_price_at_end), 4)],
                ])

    # ── Confidence: actual price-fan adherence ────────────────────────────
    adherence = _price_fan_adherence(close, anchor_low["index"], anchor_low["price"], slope_1x1)
    consistency = _trend_consistency(close, anchor_low["index"], anchor_low["price"], slope_1x1)
    # Blend: 50% adherence + 30% consistency + 20% base structure
    confidence = float(np.clip(adherence * 50 + consistency * 30 + 20, 0, 100))

    # ── Status ────────────────────────────────────────────────────────────
    # Gann fan angles are structural reference lines, not invalidatable patterns.
    # "FAILED" would hide the fan entirely — use FORMING instead when price is
    # below the 1x2 (slow bullish) fan.  Only mark FAILED if price collapses
    # more than 15% below the 1x2 line, which would genuinely break the structure.
    if last_close <= invalidation * 0.85:
        status = "FAILED"
    elif last_close >= target:
        status = "COMPLETED"
    elif last_close > breakout:
        status = "BREAKOUT"
    elif last_close >= breakout * 0.995:
        status = "READY"
    else:
        status = "FORMING"

    result.update({
        "status": status,
        "breakout_level": round(breakout, 4),
        "target": round(target, 4),
        "invalidation_level": round(invalidation, 4),
        "confidence": round(confidence, 2),
        "points": [
            [int(anchor_low["index"]), float(anchor_low["price"])],
            [int(pivot_high["index"]), float(pivot_high["price"])],
            [int(last_idx), round(float(gann_1x1), 4)],
        ],
        "overlay_lines": overlay_lines,
        "fan_lines": fan_lines,
        "retracement_levels": retracement_levels,
        "time_cycles": time_cycles,
        "square_of_9": square_of_9,
        # Bearish fan from most recent swing high (rendered separately in frontend)
        "bearish_fan_lines":    bearish_fan_lines,
        "bearish_overlay_lines": bearish_overlay_lines,
        # Direction is determined by anchor type, not by the close-vs-1x1 comparison.
        # This detector always anchors from a swing LOW and projects fan lines upward,
        # so the structure is inherently bullish.  Price trading below the 1x1 fan line
        # (but above the anchor) means the pattern is FORMING, not bearish.
        # A bearish Gann setup would require anchoring from a swing HIGH — not
        # implemented here, so direction is always "bullish" for this detector.
        "direction": "bullish",
    })
    return result
