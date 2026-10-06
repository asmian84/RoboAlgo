"""Inverse Cup & Handle detector — bearish reversal.

Mirror of Cup & Handle: an inverted-U (dome) followed by a small
downward handle, then breakdown below the rim line.

Exposes:
    detect(symbol: str, df: pd.DataFrame) -> dict
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
    """Detect an Inverse Cup & Handle pattern (bearish reversal).

    Parameters
    ----------
    symbol : str
        Ticker symbol (used for logging only).
    price_data : pd.DataFrame
        OHLCV DataFrame sorted oldest-first with columns:
        date, open, high, low, close, volume.

    Returns
    -------
    dict
        Pattern dict following the RoboAlgo service contract.
    """
    result = base_result("Inverse Cup & Handle")
    if price_data is None or len(price_data) < 90:
        return result

    try:
        df = price_data.sort_values("date").reset_index(drop=True).copy()
        adaptive_mm = compute_adaptive_minimum_move(df)
        swings = detect_swings(df, minimum_move=adaptive_mm)
        highs = swings["swing_highs"]
        lows = swings["swing_lows"]
        if len(lows) < 3 or len(highs) < 2:
            return result

        # ── Sliding window scan: try pairs of lows for best inverted cup ───
        best_cup: dict[str, Any] | None = None
        best_cup_score = -1.0

        scan_lows = lows[-6:] if len(lows) >= 6 else lows
        for li in range(len(scan_lows) - 1):
            for ri in range(li + 1, len(scan_lows)):
                left_l = scan_lows[li]
                right_l = scan_lows[ri]
                # Dome highs between left and right rim
                dome_highs = [h for h in highs if left_l["index"] < h["index"] < right_l["index"]]
                if not dome_highs:
                    continue
                dome_peak = max(dome_highs, key=lambda p: p["price"])

                rim_sim = abs(left_l["price"] - right_l["price"]) / max(left_l["price"], 1e-9)
                dome_height = (dome_peak["price"] - max(left_l["price"], right_l["price"])) / max(
                    max(left_l["price"], right_l["price"]), 1e-9
                )
                if rim_sim > 0.10 or dome_height < 0.06:
                    continue

                # Handle highs after right rim (small bounce)
                handle_highs = [h for h in highs if h["index"] > right_l["index"]]
                if not handle_highs:
                    continue
                handle_high = handle_highs[-1]
                handle_ret = (handle_high["price"] - right_l["price"]) / max(
                    dome_peak["price"] - right_l["price"], 1e-9
                )
                if handle_ret > 0.50:
                    continue

                cup_score = dome_height * 60 - rim_sim * 120 - handle_ret * 40
                if cup_score > best_cup_score:
                    best_cup_score = cup_score
                    best_cup = {
                        "left": left_l,
                        "right": right_l,
                        "dome_peak": dome_peak,
                        "handle_high": handle_high,
                        "rim_similarity": rim_sim,
                        "dome_height": dome_height,
                        "handle_retrace": handle_ret,
                    }

        if best_cup is None:
            return result

        left = best_cup["left"]
        right = best_cup["right"]
        dome_peak = best_cup["dome_peak"]
        handle_high = best_cup["handle_high"]
        rim_similarity = best_cup["rim_similarity"]
        dome_height = best_cup["dome_height"]
        handle_retrace = best_cup["handle_retrace"]

        # Breakdown below rim; target is rim minus the dome height
        breakout = float(min(left["price"], right["price"]))
        invalidation = float(handle_high["price"])
        target = float(breakout - (dome_peak["price"] - breakout))

        structure_quality = max(
            0.0, min(100.0, 75 - rim_similarity * 120 + dome_height * 60 - handle_retrace * 40)
        )
        probability = composite_probability(
            structure_quality=structure_quality,
            volume=volume_confirmation(df),
            liquidity=liquidity_alignment_score(df, breakout),
            regime=market_regime_score(df),
            momentum=momentum_score(df),
        )
        status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)

        last_idx = len(df) - 1

        # ── Dome arc: smooth inverted-U with cosine interpolation ──────────
        import math

        def _dome_arc_segments(p0_idx, p0_price, p1_idx, p1_price, n_steps=10):
            segs = []
            for k in range(n_steps + 1):
                t = k / n_steps
                t_smooth = (1 - math.cos(t * math.pi)) / 2
                idx = int(round(p0_idx + (p1_idx - p0_idx) * t))
                price = p0_price + (p1_price - p0_price) * t_smooth
                segs.append([idx, round(price, 4)])
            return segs

        left_arc = _dome_arc_segments(
            left["index"], left["price"], dome_peak["index"], dome_peak["price"], 10
        )
        right_arc = _dome_arc_segments(
            dome_peak["index"], dome_peak["price"], right["index"], right["price"], 10
        )
        dome_pts = left_arc + right_arc[1:]

        overlay_lines = []
        # Dome arc as consecutive segments
        for i in range(len(dome_pts) - 1):
            overlay_lines.append([dome_pts[i], dome_pts[i + 1]])
        # Handle bounce: right_rim -> handle_high
        overlay_lines.append(
            [[right["index"], right["price"]], [handle_high["index"], handle_high["price"]]]
        )
        # Handle drop: handle_high -> last_idx at breakdown level
        overlay_lines.append(
            [[handle_high["index"], handle_high["price"]], [last_idx, breakout]]
        )
        # Neckline: horizontal at breakdown level
        overlay_lines.append([[left["index"], breakout], [last_idx, breakout]])
        # Measured move: breakdown -> target
        overlay_lines.append([[last_idx, breakout], [last_idx, target]])

        n_arc_segs = len(dome_pts) - 1
        overlay_line_roles = (
            ["dome_arc"] * n_arc_segs
            + ["handle_bounce", "handle_drop", "neckline", "measured_move"]
        )

        fill_zone = {
            "upper": 0,
            "lower": n_arc_segs - 1,
            "color": "#ef4444",
            "opacity": 0.08,
        }

        result.update(
            {
                "status": status,
                "direction": "bearish",
                "breakout_level": round(breakout, 4),
                "invalidation_level": round(invalidation, 4),
                "projected_target": round(target, 4),
                "probability": round(probability, 2),
                "confidence": round(probability, 2),
                "points": [
                    [left["index"], left["price"]],
                    [dome_peak["index"], dome_peak["price"]],
                    [right["index"], right["price"]],
                    [handle_high["index"], handle_high["price"]],
                ],
                "overlay_lines": overlay_lines,
                "overlay_line_roles": overlay_line_roles,
                "fill_zone": fill_zone,
                "point_labels": ["RIM L", "DOME", "RIM R", "HANDLE"],
            }
        )
        return result

    except Exception:
        return result
