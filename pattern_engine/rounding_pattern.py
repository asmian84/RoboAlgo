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
    """Return [Rounding Bottom, Rounding Top] detections."""
    rb = base_result("Rounding Bottom")
    rt = base_result("Rounding Top")
    if price_data is None or len(price_data) < _LOOKBACK:
        return [rb, rt]

    df  = price_data.sort_values("date").reset_index(drop=True).copy()
    win = df["close"].astype(float).tail(_LOOKBACK).reset_index(drop=True)
    y   = win.values

    a, b, c, r2 = _fit_parabola(y)
    if r2 < _MIN_R2:
        return [rb, rt]

    n        = len(y)
    vertex_x = -b / (2.0 * a) if abs(a) > 1e-12 else n / 2.0
    # Vertex must be centred (not at the edges) for a proper arc
    if not (n * 0.15 < vertex_x < n * 0.85):
        return [rb, rt]

    start_global = len(df) - _LOOKBACK
    last_idx     = len(df) - 1
    neckline     = float((y[0] + y[-1]) / 2.0)

    # Arc overlay: 6 segments across the window
    arc_x = np.linspace(0, n - 1, 7)
    arc_y = a * arc_x ** 2 + b * arc_x + c
    arc_global = [int(start_global + xi) for xi in arc_x]
    arc_lines = [
        [[arc_global[i], float(arc_y[i])], [arc_global[i + 1], float(arc_y[i + 1])]]
        for i in range(len(arc_global) - 1)
    ]
    neckline_seg = [[start_global, neckline], [last_idx, neckline]]
    overlay_lines = arc_lines + [neckline_seg]
    roles = ["arc"] * 6 + ["neckline"]

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
        status = status_from_levels(df["close"], breakout, invalidation, target, bullish=True)
        rb.update({
            "pattern_name": "Rounding Bottom", "status": status,
            "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4), "confidence": round(prob, 2),
            "probability": round(prob, 2), "direction": "bullish",
            "points": [[arc_global[i], float(arc_y[i])] for i in range(0, 7, 2)],
            "overlay_lines": overlay_lines, "overlay_line_roles": roles,
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
        status = status_from_levels(df["close"], breakout, invalidation, target, bullish=False)
        rt.update({
            "pattern_name": "Rounding Top", "status": status,
            "breakout_level": round(breakout, 4), "invalidation_level": round(invalidation, 4),
            "projected_target": round(target, 4), "confidence": round(prob, 2),
            "probability": round(prob, 2), "direction": "bearish",
            "points": [[arc_global[i], float(arc_y[i])] for i in range(0, 7, 2)],
            "overlay_lines": overlay_lines, "overlay_line_roles": roles,
        })

    return [rb, rt]
