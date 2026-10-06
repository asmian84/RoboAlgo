"""Megaphone / Broadening Formation detector.

Ascending Broadening Pattern (TradingView style):
  - Alternating swing highs (1, 3, 5) rising AND swing lows (2, 4) falling
  - Upper trendline drawn through swing HIGH pivots (touches actual candles)
  - Lower trendline drawn through swing LOW pivots (touches actual candles)
  - Lines diverge — the hallmark of a broadening/megaphone pattern
  - Numbered pivot labels (1, 2, 3, 4, 5) for professional look
  - Lines extend forward beyond current price

Detection: fit OLS through last 3+ swing highs AND 3+ swing lows.
Both lines must be expanding (highs rising, lows falling or at least
diverging more than the convergence threshold).
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


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    result = base_result("Megaphone")
    if price_data is None or len(price_data) < 70:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)

    best = None
    best_prob = 0.0

    for scale in [1.0, 2.0, 3.5]:
        mm = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        highs = swings["swing_highs"][-5:]
        lows = swings["swing_lows"][-5:]
        if len(highs) < 2 or len(lows) < 2:
            continue

        xh = np.array([p["index"] for p in highs], dtype=float)
        yh = np.array([p["price"] for p in highs], dtype=float)
        xl = np.array([p["index"] for p in lows], dtype=float)
        yl = np.array([p["price"] for p in lows], dtype=float)
        h_slope, h_int = np.polyfit(xh, yh, 1)
        l_slope, l_int = np.polyfit(xl, yl, 1)

        # Megaphone: expanding — highs rising, lows falling
        if h_slope <= 0 or l_slope >= 0:
            continue

        last_idx = len(df) - 1
        upper = float(h_slope * last_idx + h_int)
        lower = float(l_slope * last_idx + l_int)
        width_now = max(upper - lower, 0.0)

        early_idx = max(int(np.mean([p["index"] for p in highs[:1] + lows[:1]])), 1)
        early_upper = float(h_slope * early_idx + h_int)
        early_lower = float(l_slope * early_idx + l_int)
        width_early = max(early_upper - early_lower, 1e-9)
        if width_now <= width_early * 1.15:
            continue

        breakout = upper
        invalidation = lower
        target = float(breakout + width_now * 0.75)
        sq = float(np.clip(55 + (width_now / width_early) * 20, 0, 100))
        prob = composite_probability(
            sq, volume_confirmation(df),
            liquidity_alignment_score(df, breakout),
            market_regime_score(df), momentum_score(df),
        )

        if prob > best_prob:
            best_prob = prob
            best = (highs, lows, h_slope, h_int, l_slope, l_int,
                    breakout, invalidation, target, prob)

    if best is None:
        return result

    highs, lows, h_slope, h_int, l_slope, l_int, breakout, invalidation, target, prob = best
    last_idx = len(df) - 1
    fwd_idx = last_idx + 20  # extend lines 20 bars into the future

    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)

    # ── Build overlay: lines through ACTUAL swing points ──────────────────
    # Upper trendline: from first swing high to last, extended forward
    first_h = highs[0]
    upper_start_price = float(h_slope * first_h["index"] + h_int)
    upper_end_price = float(h_slope * fwd_idx + h_int)

    # Lower trendline: from first swing low to last, extended forward
    first_l = lows[0]
    lower_start_price = float(l_slope * first_l["index"] + l_int)
    lower_end_price = float(l_slope * fwd_idx + l_int)

    first_idx = int(min(first_h["index"], first_l["index"]))

    overlay_lines = [
        # Upper trendline — solid, bold, through swing highs, extended forward
        [[first_idx, float(h_slope * first_idx + h_int)], [fwd_idx, upper_end_price]],
        # Lower trendline — solid, bold, through swing lows, extended forward
        [[first_idx, float(l_slope * first_idx + l_int)], [fwd_idx, lower_end_price]],
        # Measured move: breakout → target
        [[last_idx, round(breakout, 4)], [last_idx, round(target, 4)]],
    ]

    # ── Build numbered pivot points: interleave highs and lows chronologically ──
    all_pivots = []
    for p in highs:
        all_pivots.append({"index": int(p["index"]), "price": float(p["price"]), "type": "H"})
    for p in lows:
        all_pivots.append({"index": int(p["index"]), "price": float(p["price"]), "type": "L"})
    all_pivots.sort(key=lambda x: x["index"])
    # Take up to 5 pivots for clean numbering
    all_pivots = all_pivots[:7]

    points = [[p["index"], p["price"]] for p in all_pivots]
    point_labels = [str(i + 1) for i in range(len(all_pivots))]

    result.update({
        "pattern_name":       "Broadening Formation",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "confidence":         round(prob, 2),
        "probability":        round(prob, 2),
        "direction":          "neutral",
        "points":             points,
        "point_labels":       point_labels,
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "measured_move"],
        "fill_zone": {"upper": 0, "lower": 1, "color": "#818cf8", "opacity": 0.06},
    })
    return result
