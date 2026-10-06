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


def _find_best_cup(highs, lows, rim_max=0.30, dep_min=0.06, handle_max=0.50):
    """Score all high-pairs for cup + handle geometry. Returns best or None."""
    best_cup: dict[str, Any] | None = None
    best_cup_score = -1.0

    # Use recent highs + always include the global high (covers macro cups)
    scan_highs = highs[-12:] if len(highs) >= 12 else highs
    if highs:
        global_high = max(highs, key=lambda h: h["price"])
        if global_high not in scan_highs:
            scan_highs = [global_high] + list(scan_highs)
    for li in range(len(scan_highs) - 1):
        for ri in range(li + 1, len(scan_highs)):
            left_h = scan_highs[li]
            right_h = scan_highs[ri]
            cup_lows_inner = [l for l in lows if left_h["index"] < l["index"] < right_h["index"]]
            if not cup_lows_inner:
                continue
            cup_low_inner = min(cup_lows_inner, key=lambda p: p["price"])
            rim_sim = abs(left_h["price"] - right_h["price"]) / max(left_h["price"], 1e-9)
            cup_dep = (min(left_h["price"], right_h["price"]) - cup_low_inner["price"]) / max(min(left_h["price"], right_h["price"]), 1e-9)
            if rim_sim > rim_max or cup_dep < dep_min:
                continue
            handle_lows_inner = [l for l in lows if l["index"] > right_h["index"]]
            if not handle_lows_inner:
                continue
            handle_low_inner = handle_lows_inner[-1]
            handle_ret = (right_h["price"] - handle_low_inner["price"]) / max(right_h["price"] - cup_low_inner["price"], 1e-9)
            if handle_ret > handle_max:
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
    return best_cup


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    result = base_result("Cup & Handle")
    if price_data is None or len(price_data) < 90:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()

    # ── Multi-scale swing detection ─────────────────────────────────────
    # Scale 1: adaptive (fine-grained) — catches small/medium cups
    # Scale 2: 3× adaptive — catches large structural cups spanning the chart
    adaptive_mm = compute_adaptive_minimum_move(df)
    best_cup: dict[str, Any] | None = None

    for mm_mult in [1.0, 3.0, 6.0]:
        mm = max(adaptive_mm * mm_mult, 0.01)
        swings = detect_swings(df, minimum_move=mm)
        highs = swings["swing_highs"]
        lows = swings["swing_lows"]
        if len(highs) < 3 or len(lows) < 2:
            continue
        candidate = _find_best_cup(highs, lows)
        if candidate is not None:
            # Prefer deeper cups (more significant patterns)
            if best_cup is None or candidate["cup_depth"] > best_cup["cup_depth"]:
                best_cup = candidate

    # ── Forming-cup check: right rim still climbing toward left rim ─────
    # Check for a large rounding bottom where price has recovered >30% of
    # the drop from a major high. Preferred over a small completed cup if
    # the forming cup's absolute price range is ≥2× larger.
    forming = False
    if True:  # always check, compare against best_cup
        # Use 1× scale (most pivots) — we need the global high
        swings = detect_swings(df, minimum_move=adaptive_mm)
        highs = swings["swing_highs"]
        lows = swings["swing_lows"]
        if highs and lows:
            global_high = max(highs, key=lambda h: h["price"])
            # Cup bottom: lowest low after the global high
            cup_lows = [l for l in lows if l["index"] > global_high["index"]]
            if cup_lows:
                cup_low_pt = min(cup_lows, key=lambda l: l["price"])
                # Right rim: highest high after the cup bottom
                right_highs = [h for h in highs if h["index"] > cup_low_pt["index"]]
                if right_highs:
                    right_rim = max(right_highs, key=lambda h: h["price"])
                    drop = global_high["price"] - cup_low_pt["price"]
                    recovery = right_rim["price"] - cup_low_pt["price"]
                    recovery_pct = recovery / drop if drop > 0 else 0
                    cup_dep = drop / global_high["price"] if global_high["price"] > 0 else 0
                    # Accept if: deep cup (>30%) and meaningful recovery (>30% of drop)
                    if cup_dep > 0.30 and recovery_pct > 0.30:
                        # Prefer over existing best_cup if ≥2× larger absolute range
                        existing_range = (best_cup["left"]["price"] - best_cup["cup_low"]["price"]) if best_cup else 0
                        forming_range = drop
                        if best_cup is None or forming_range >= existing_range * 2:
                            recent_lows = [l for l in lows if l["index"] > right_rim["index"]]
                            handle_pt = recent_lows[-1] if recent_lows else right_rim
                            rim_sim = abs(global_high["price"] - right_rim["price"]) / max(global_high["price"], 1e-9)
                            handle_ret = (right_rim["price"] - handle_pt["price"]) / max(right_rim["price"] - cup_low_pt["price"], 1e-9) if recent_lows else 0
                            best_cup = {
                                "left": global_high, "right": right_rim,
                                "cup_low": cup_low_pt, "handle_low": handle_pt,
                                "rim_similarity": rim_sim, "cup_depth": cup_dep,
                                "handle_retrace": handle_ret,
                            }
                            forming = True

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
    if forming:
        status = "FORMING"
    else:
        status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    last_idx = len(df) - 1

    # ── Cup arc: smooth U-shape with many segments for professional look ────
    # Uses cosine interpolation for a naturally rounded curve
    def _cup_arc_segments(p0_idx, p0_price, p1_idx, p1_price, n_steps=10):
        """Return intermediate [idx, price] points along a smooth rounded arc."""
        import math
        segs = []
        for k in range(n_steps + 1):
            t = k / n_steps
            # Cosine interpolation: smooth S-curve that creates natural rounding
            t_smooth = (1 - math.cos(t * math.pi)) / 2
            idx   = int(round(p0_idx + (p1_idx - p0_idx) * t))
            price = p0_price + (p1_price - p0_price) * t_smooth
            segs.append([idx, round(price, 4)])
        return segs

    left_arc  = _cup_arc_segments(left["index"], left["price"], cup_low["index"], cup_low["price"], 10)
    right_arc = _cup_arc_segments(cup_low["index"], cup_low["price"], right["index"], right["price"], 10)
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
    # Neckline: horizontal at breakout (rim) level from left_rim to last_idx
    overlay_lines.append([[left["index"], breakout], [last_idx, breakout]])
    # Measured move: vertical arrow from breakout to target at last_idx
    overlay_lines.append([[last_idx, breakout], [last_idx, target]])

    # Roles: "cup_arc" for each arc segment, then handle parts, neckline, measured move
    n_arc_segs = len(cup_pts) - 1
    overlay_line_roles = (
        ["cup_arc"] * n_arc_segs
        + ["handle_pullback", "handle_recovery", "neckline", "measured_move"]
    )

    # Fill zone: shade the cup interior (first arc seg to last arc seg)
    fill_zone = {
        "upper": n_arc_segs - 1,   # last arc segment (near right rim)
        "lower": 0,                  # first arc segment (near left rim)
        "color": "#22c55e",
        "opacity": 0.08,
    }

    result.update(
        {
            "status": status,
            "direction": "bullish",
            "breakout_level": round(breakout, 4),
            "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4),
            "probability": round(probability, 2),
            "confidence": round(probability, 2),
            "points": [
                [left["index"], left["price"]],
                [cup_low["index"], cup_low["price"]],
                [right["index"], right["price"]],
                [handle_low["index"], handle_low["price"]],
            ],
            "overlay_lines": overlay_lines,
            "overlay_line_roles": overlay_line_roles,
            "fill_zone": fill_zone,
            "point_labels": ["RIM L", "BASE", "RIM R", "HANDLE"],
            # Named key points for frontend labeling
            "cup_left_price": round(float(left["price"]), 4),
            "cup_right_price": round(float(right["price"]), 4),
            "cup_bottom_price": round(float(cup_low["price"]), 4),
        }
    )
    return result

