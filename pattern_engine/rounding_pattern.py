"""Rounding Bottom (Saucer) and Rounding Top detector.

Rounding Bottom:
  - Price forms a smooth U-shaped curve over many bars
  - Gradual deceleration of downtrend → flat base → gradual acceleration upward
  - Bullish breakout above the neckline (rim level)
  - Target = neckline + (neckline − cup_low)

Rounding Top:
  - Inverse dome / n-shape
  - Bearish breakdown below neckline
  - Target = neckline − (cup_high − neckline)

Detection: fit a parabola (y = ax² + bx + c) to the last _LOOKBACK bars.
  - a > 0 and vertex well-centred → Rounding Bottom
  - a < 0 and vertex well-centred → Rounding Top
  - R² goodness-of-fit determines confidence.
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

_LOOKBACK = 60   # bars for parabola window
_MIN_R2   = 0.55  # minimum R² to qualify


def _fit_parabola(y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit y = ax² + bx + c; return (a, b, c, r²)."""
    n = len(y)
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return float(a), float(b), float(c), float(r2)


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Rounding Bottom, Rounding Top] detections.

    Multi-scale: tries multiple lookback windows (60, 120, full chart).
    Larger arcs that span the whole chart get a span bonus.
    """
    rb = base_result("Rounding Bottom")
    rt = base_result("Rounding Top")
    if price_data is None or len(price_data) < _LOOKBACK:
        return [rb, rt]

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    total_bars = len(df)

    # Try multiple lookback windows — full chart arc beats a small one
    lookbacks = [_LOOKBACK]
    if total_bars >= 120:
        lookbacks.append(120)
    if total_bars >= 180:
        lookbacks.append(total_bars)  # full chart

    best_rb, best_rt = rb, rt
    best_rb_conf, best_rt_conf = 0.0, 0.0

    for lb in lookbacks:
        rb_cand, rt_cand = _detect_rounding_at_lookback(df, lb, total_bars)
        rc = float(rb_cand.get("confidence", 0.0))
        tc = float(rt_cand.get("confidence", 0.0))
        if rc > best_rb_conf:
            best_rb_conf = rc
            best_rb = rb_cand
        if tc > best_rt_conf:
            best_rt_conf = tc
            best_rt = rt_cand

    return [best_rb, best_rt]


def _detect_rounding_at_lookback(
    df: pd.DataFrame, lookback: int, total_bars: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect rounding patterns at a specific lookback window."""
    rb = base_result("Rounding Bottom")
    rt = base_result("Rounding Top")

    win = df["close"].astype(float).tail(lookback).reset_index(drop=True)
    y   = win.values

    a, b, c, r2 = _fit_parabola(y)
    if r2 < _MIN_R2:
        return [rb, rt]

    n        = len(y)
    vertex_x = -b / (2.0 * a) if abs(a) > 1e-12 else n / 2.0
    # Vertex must be centred (not at the edges) for a proper arc
    if not (n * 0.15 < vertex_x < n * 0.85):
        return [rb, rt]

    start_global = len(df) - lookback
    last_idx     = len(df) - 1
    neckline     = float((y[0] + y[-1]) / 2.0)

    # Arc overlay: 12 segments across the window for smooth professional curve
    fwd_idx = last_idx + 15
    n_arc_pts = 13
    arc_x = np.linspace(0, n - 1, n_arc_pts)
    arc_y = a * arc_x ** 2 + b * arc_x + c
    arc_global = [int(start_global + xi) for xi in arc_x]
    arc_lines = [
        [[arc_global[i], round(float(arc_y[i]), 4)], [arc_global[i + 1], round(float(arc_y[i + 1]), 4)]]
        for i in range(len(arc_global) - 1)
    ]
    n_arc_segs = len(arc_lines)
    # Neckline: horizontal, extended forward
    neckline_seg = [[start_global, round(neckline, 4)], [fwd_idx, round(neckline, 4)]]
    overlay_lines = arc_lines + [neckline_seg]
    roles = ["arc"] * n_arc_segs + ["neckline"]

    base_conf = float(np.clip(r2 * 100, 0, 100))

    if a > 0:
        # Rounding Bottom
        cup_low    = float(y[int(round(vertex_x))])
        breakout   = neckline
        target     = neckline + (neckline - cup_low)
        invalidation = float(min(y))
        prob = composite_probability(base_conf, volume_confirmation(df),
                                     liquidity_alignment_score(df, breakout),
                                     market_regime_score(df), momentum_score(df))
        # Span bonus: arcs covering more of the chart are more significant
        span_ratio = lookback / max(total_bars, 1)
        prob = min(prob + span_ratio * 15, 99.0)
        status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)
        # Add measured move line
        rb_lines = overlay_lines + [[[last_idx, round(breakout, 4)], [last_idx, round(target, 4)]]]
        rb_roles = roles + ["measured_move"]
        rb.update({
            "pattern_name": "Rounding Bottom", "status": status,
            "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4), "confidence": round(prob, 2),
            "probability": round(prob, 2), "direction": "bullish",
            "points": [[arc_global[i], float(arc_y[i])] for i in range(0, 7, 2)],
            "point_labels": ["START", "DIP", "MID", "RIM"],
            "overlay_lines": rb_lines, "overlay_line_roles": rb_roles,
        })

    if a < 0:
        # Rounding Top
        cup_high   = float(y[int(round(vertex_x))])
        breakout   = neckline
        target     = neckline - (cup_high - neckline)
        invalidation = float(max(y))
        prob = composite_probability(base_conf, volume_confirmation(df),
                                     liquidity_alignment_score(df, breakout),
                                     market_regime_score(df), momentum_score(df))
        span_ratio = lookback / max(total_bars, 1)
        prob = min(prob + span_ratio * 15, 99.0)
        status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)
        # Add measured move line
        rt_lines = overlay_lines + [[[last_idx, round(breakout, 4)], [last_idx, round(target, 4)]]]
        rt_roles = roles + ["measured_move"]
        rt.update({
            "pattern_name": "Rounding Top", "status": status,
            "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4), "confidence": round(prob, 2),
            "probability": round(prob, 2), "direction": "bearish",
            "points": [[arc_global[i], float(arc_y[i])] for i in range(0, 7, 2)],
            "point_labels": ["START", "PEAK", "MID", "RIM"],
            "overlay_lines": rt_lines, "overlay_line_roles": rt_roles,
        })

    return [rb, rt]
