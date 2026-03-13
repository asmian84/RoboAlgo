"""Cup & Handle detector using swing highs/lows."""

from __future__ import annotations

from typing import Any

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
    result = base_result("Cup & Handle")
    if price_data is None or len(price_data) < 90:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)
    highs = swings["swing_highs"]
    lows = swings["swing_lows"]
    if len(highs) < 3 or len(lows) < 2:
        return result

    # ── Sliding window scan: try all pairs of highs for best cup ─────────
    best_cup: dict[str, Any] | None = None
    best_cup_score = -1.0

    scan_highs = highs[-6:] if len(highs) >= 6 else highs
    for li in range(len(scan_highs) - 1):
        for ri in range(li + 1, len(scan_highs)):
            left_h = scan_highs[li]
            right_h = scan_highs[ri]
            # Cup lows between left and right rim
            cup_lows_inner = [l for l in lows if left_h["index"] < l["index"] < right_h["index"]]
            if not cup_lows_inner:
                continue
            cup_low_inner = min(cup_lows_inner, key=lambda p: p["price"])
            rim_sim = abs(left_h["price"] - right_h["price"]) / max(left_h["price"], 1e-9)
            cup_dep = (min(left_h["price"], right_h["price"]) - cup_low_inner["price"]) / max(min(left_h["price"], right_h["price"]), 1e-9)
            if rim_sim > 0.10 or cup_dep < 0.06:
                continue
            # Handle lows after right rim
            handle_lows_inner = [l for l in lows if l["index"] > right_h["index"]]
            if not handle_lows_inner:
                continue
            handle_low_inner = handle_lows_inner[-1]
            handle_ret = (right_h["price"] - handle_low_inner["price"]) / max(right_h["price"] - cup_low_inner["price"], 1e-9)
            if handle_ret > 0.5:
                continue
            cup_score = cup_dep * 60 - rim_sim * 120 - handle_ret * 40
            if cup_score > best_cup_score:
                best_cup_score = cup_score
                best_cup = {
                    "left": left_h, "right": right_h,
                    "cup_low": cup_low_inner, "handle_low": handle_low_inner,
                    "rim_similarity": rim_sim, "cup_depth": cup_dep,
                    "handle_retrace": handle_ret,
                }

    if best_cup is None:
        return result

    left = best_cup["left"]
    right = best_cup["right"]
    cup_low = best_cup["cup_low"]
    handle_low = best_cup["handle_low"]
    rim_similarity = best_cup["rim_similarity"]
    cup_depth = best_cup["cup_depth"]
    handle_retrace = best_cup["handle_retrace"]

    breakout = float(max(left["price"], right["price"]))
    invalidation = float(handle_low["price"])
    target = float(breakout + (breakout - cup_low["price"]))

    structure_quality = max(0.0, min(100.0, 75 - rim_similarity * 120 + cup_depth * 60 - handle_retrace * 40))
    probability = composite_probability(
        structure_quality=structure_quality,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    last_idx = len(df) - 1

    # ── Cup arc: approximate smooth U with intermediate points ─────────────
    # Left wall: left_rim → cup_bottom (descending, parabolic)
    # Right wall: cup_bottom → right_rim (ascending, parabolic)
    # Handle: right_rim → handle_low (pullback) → last bar at breakout (recovery)
    def _cup_arc_segments(p0_idx, p0_price, p1_idx, p1_price, n_steps=4):
        """Return n_steps intermediate [idx, price] points along a parabolic arc."""
        segs = []
        for k in range(n_steps + 1):
            t = k / n_steps
            # Parabolic weighting: stays near high longer then drops/rises quickly
            t_para = t * t if p0_price > p1_price else 1 - (1 - t) * (1 - t)
            idx   = int(round(p0_idx + (p1_idx - p0_idx) * t))
            price = p0_price + (p1_price - p0_price) * t_para
            segs.append([idx, round(price, 4)])
        return segs

    left_arc  = _cup_arc_segments(left["index"], left["price"], cup_low["index"], cup_low["price"])
    right_arc = _cup_arc_segments(cup_low["index"], cup_low["price"], right["index"], right["price"])
    cup_pts   = left_arc + right_arc[1:]   # merge, avoid duplicate cup_low

    # Build overlay_lines from the cup arc + handle segments
    overlay_lines = []
    # Cup arc as consecutive segments
    for i in range(len(cup_pts) - 1):
        overlay_lines.append([cup_pts[i], cup_pts[i + 1]])
    # Handle pullback: right_rim → handle_low
    overlay_lines.append([[right["index"], right["price"]], [handle_low["index"], handle_low["price"]]])
    # Handle recovery: handle_low → last_idx at breakout level (shows where price is heading)
    overlay_lines.append([[handle_low["index"], handle_low["price"]], [last_idx, breakout]])

    # Roles: "cup_arc" for each arc segment, then handle parts
    n_arc_segs = len(cup_pts) - 1
    overlay_line_roles = (
        ["cup_arc"] * n_arc_segs
        + ["handle_pullback", "handle_recovery"]
    )

    result.update(
        {
            "status": status,
            "breakout_level": round(breakout, 4),
            "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4),
            "probability": round(probability, 2),
            "points": [
                [left["index"], left["price"]],
                [cup_low["index"], cup_low["price"]],
                [right["index"], right["price"]],
                [handle_low["index"], handle_low["price"]],
            ],
            "overlay_lines": overlay_lines,
            "overlay_line_roles": overlay_line_roles,
            "point_labels": ["Rim L", "Base", "Rim R", "Handle"],
            # Named key points for frontend labeling
            "cup_left_price": round(float(left["price"]), 4),
            "cup_right_price": round(float(right["price"]), 4),
            "cup_bottom_price": round(float(cup_low["price"]), 4),
        }
    )
    return result

