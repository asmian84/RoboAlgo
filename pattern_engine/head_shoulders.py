"""Head & Shoulders and Inverse Head & Shoulders detector.

Head & Shoulders (bearish reversal):
  - 3 swing peaks: Left Shoulder → Head (highest) → Right Shoulder
  - Neckline: line connecting the troughs between LS↔H and H↔RS
  - Breakdown below neckline → target = neckline − (head − neckline)

Inverse Head & Shoulders (bullish reversal):
  - 3 swing troughs: Left Shoulder → Head (lowest) → Right Shoulder
  - Neckline: line connecting the peaks between LS↔H and H↔RS
  - Breakout above neckline → target = neckline + (neckline − head)
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


def _detect_hs(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Detect Head & Shoulders (bearish reversal)."""
    result = base_result("Head & Shoulders")
    if len(highs) < 3 or len(lows) < 2:
        return result

    last_idx = len(df) - 1
    best: tuple | None = None

    # Slide a window over last 7 highs to find best H&S formation
    candidates = highs[-7:]
    for i in range(len(candidates) - 2):
        ls, h, rs = candidates[i], candidates[i + 1], candidates[i + 2]

        # Head must be the tallest peak
        if h["price"] <= ls["price"] or h["price"] <= rs["price"]:
            continue

        # Shoulders should be roughly symmetric (within 30%)
        shoulder_sym = abs(ls["price"] - rs["price"]) / max(h["price"], 1e-9)
        if shoulder_sym > 0.30:
            continue

        # Right shoulder must be to the right of head
        if rs["index"] <= h["index"] or h["index"] <= ls["index"]:
            continue

        # Find the deepest trough between LS–H and H–RS
        t1_cands = [l for l in lows if ls["index"] < l["index"] < h["index"]]
        t2_cands = [l for l in lows if h["index"] < l["index"] < rs["index"]]
        if not t1_cands or not t2_cands:
            continue
        t1 = min(t1_cands, key=lambda x: x["price"])
        t2 = min(t2_cands, key=lambda x: x["price"])

        # Neckline through the two troughs
        nx1, ny1 = float(t1["index"]), float(t1["price"])
        nx2, ny2 = float(t2["index"]), float(t2["price"])
        if nx2 == nx1:
            continue
        neck_slope = (ny2 - ny1) / (nx2 - nx1)
        neck_int = ny1 - neck_slope * nx1

        head_on_neck = neck_slope * h["index"] + neck_int
        head_height = h["price"] - head_on_neck
        if head_height <= 0:
            continue

        quality = shoulder_sym  # lower = more symmetric = better
        if best is None or quality < best[0]:
            best = (quality, ls, h, rs, t1, t2, neck_slope, neck_int, head_height)

    if best is None:
        return result

    quality, ls, h, rs, t1, t2, neck_slope, neck_int, head_height = best
    neckline_now = neck_slope * last_idx + neck_int
    breakout = float(neckline_now)          # bearish: break BELOW neckline
    target = float(neckline_now - head_height)
    invalidation = float(h["price"])        # price above head = pattern failed

    structure_quality = float(np.clip(75 - (quality / 0.30) * 30, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

    neck_left_pt  = float(neck_slope * t1["index"] + neck_int)
    neck_right_pt = float(neck_slope * t2["index"] + neck_int)

    overlay_lines = [
        # LS → trough1
        [[int(ls["index"]), float(ls["price"])],  [int(t1["index"]), float(t1["price"])]],
        # trough1 → Head
        [[int(t1["index"]), float(t1["price"])],  [int(h["index"]),  float(h["price"])]],
        # Head → trough2
        [[int(h["index"]),  float(h["price"])],   [int(t2["index"]), float(t2["price"])]],
        # trough2 → RS
        [[int(t2["index"]), float(t2["price"])],  [int(rs["index"]), float(rs["price"])]],
        # RS → neckline (completion — right side breakdown leg)
        [[int(rs["index"]), float(rs["price"])],  [last_idx, float(neckline_now)]],
        # Neckline (between troughs)
        [[int(t1["index"]), neck_left_pt],         [int(t2["index"]), neck_right_pt]],
        # Neckline extension to right edge
        [[int(t2["index"]), neck_right_pt],        [last_idx, float(neckline_now)]],
    ]
    overlay_line_roles = [
        "ls_down", "head_up", "head_down", "rs_up", "rs_down", "neckline", "neckline_ext",
    ]

    result.update({
        "pattern_name":       "Head & Shoulders",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(probability, 2),
        "probability":        round(probability, 2),
        "direction":          "bearish",
        "points": [
            [int(ls["index"]), float(ls["price"])],
            [int(h["index"]),  float(h["price"])],
            [int(rs["index"]), float(rs["price"])],
            [int(t1["index"]), float(t1["price"])],
            [int(t2["index"]), float(t2["price"])],
        ],
        "point_labels":       ["LS", "H", "RS", "", ""],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": overlay_line_roles,
    })
    return result


def _detect_ihs(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Detect Inverse Head & Shoulders (bullish reversal)."""
    result = base_result("Inv. Head & Shoulders")
    if len(lows) < 3 or len(highs) < 2:
        return result

    last_idx = len(df) - 1
    best: tuple | None = None

    candidates = lows[-7:]
    for i in range(len(candidates) - 2):
        ls, h, rs = candidates[i], candidates[i + 1], candidates[i + 2]

        # Head must be the lowest trough
        if h["price"] >= ls["price"] or h["price"] >= rs["price"]:
            continue

        shoulder_sym = abs(ls["price"] - rs["price"]) / max(abs(h["price"]), 1e-9)
        if shoulder_sym > 0.30:
            continue

        if rs["index"] <= h["index"] or h["index"] <= ls["index"]:
            continue

        # Find the highest peak between LS–H and H–RS
        t1_cands = [hi for hi in highs if ls["index"] < hi["index"] < h["index"]]
        t2_cands = [hi for hi in highs if h["index"] < hi["index"] < rs["index"]]
        if not t1_cands or not t2_cands:
            continue
        t1 = max(t1_cands, key=lambda x: x["price"])
        t2 = max(t2_cands, key=lambda x: x["price"])

        nx1, ny1 = float(t1["index"]), float(t1["price"])
        nx2, ny2 = float(t2["index"]), float(t2["price"])
        if nx2 == nx1:
            continue
        neck_slope = (ny2 - ny1) / (nx2 - nx1)
        neck_int = ny1 - neck_slope * nx1

        head_on_neck = neck_slope * h["index"] + neck_int
        head_depth = head_on_neck - h["price"]
        if head_depth <= 0:
            continue

        quality = shoulder_sym
        if best is None or quality < best[0]:
            best = (quality, ls, h, rs, t1, t2, neck_slope, neck_int, head_depth)

    if best is None:
        return result

    quality, ls, h, rs, t1, t2, neck_slope, neck_int, head_depth = best
    neckline_now = neck_slope * last_idx + neck_int
    breakout = float(neckline_now)          # bullish: break ABOVE neckline
    target = float(neckline_now + head_depth)
    invalidation = float(h["price"])        # price below head = pattern failed

    structure_quality = float(np.clip(75 - (quality / 0.30) * 30, 0, 100))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    neck_left_pt  = float(neck_slope * t1["index"] + neck_int)
    neck_right_pt = float(neck_slope * t2["index"] + neck_int)

    overlay_lines = [
        [[int(ls["index"]), float(ls["price"])],  [int(t1["index"]), float(t1["price"])]],
        [[int(t1["index"]), float(t1["price"])],  [int(h["index"]),  float(h["price"])]],
        [[int(h["index"]),  float(h["price"])],   [int(t2["index"]), float(t2["price"])]],
        [[int(t2["index"]), float(t2["price"])],  [int(rs["index"]), float(rs["price"])]],
        # RS → neckline (completion — right side breakout leg)
        [[int(rs["index"]), float(rs["price"])],  [last_idx, float(neckline_now)]],
        [[int(t1["index"]), neck_left_pt],         [int(t2["index"]), neck_right_pt]],
        [[int(t2["index"]), neck_right_pt],        [last_idx, float(neckline_now)]],
    ]
    overlay_line_roles = [
        "ls_up", "head_down", "head_up", "rs_down", "rs_up", "neckline", "neckline_ext",
    ]

    result.update({
        "pattern_name":       "Inv. Head & Shoulders",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(probability, 2),
        "probability":        round(probability, 2),
        "direction":          "bullish",
        "points": [
            [int(ls["index"]), float(ls["price"])],
            [int(h["index"]),  float(h["price"])],
            [int(rs["index"]), float(rs["price"])],
            [int(t1["index"]), float(t1["price"])],
            [int(t2["index"]), float(t2["price"])],
        ],
        "point_labels":       ["LS", "H", "RS", "", ""],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": overlay_line_roles,
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Head & Shoulders, Inv. Head & Shoulders] detections."""
    if price_data is None or len(price_data) < 60:
        return [base_result("Head & Shoulders"), base_result("Inv. Head & Shoulders")]

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    return [
        _detect_hs(df, highs, lows),
        _detect_ihs(df, highs, lows),
    ]
