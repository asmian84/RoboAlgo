"""Volatility force: expansion/contraction momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volatility_force(df: pd.DataFrame) -> float:
    """Compute volatility force from -1 (contracting fast) to +1 (expanding fast).

    Measures the rate of volatility change — not direction, but energy.
    Positive = volatility expanding (supports breakouts).
    Negative = volatility contracting (supports mean reversion).
    """
    if df is None or len(df) < 30:
        return 0.0

    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values

    # ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr_short = float(np.mean(tr[-5:])) if len(tr) >= 5 else float(np.mean(tr))
    atr_long = float(np.mean(tr[-20:])) if len(tr) >= 20 else float(np.mean(tr))

    if atr_long <= 0:
        return 0.0

    # Expansion ratio
    expansion_ratio = atr_short / atr_long - 1.0  # >0 = expanding, <0 = contracting

    # Bollinger Band width change
    bb_mid = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    if len(bb_std) >= 5 and bb_mid[-1] > 0:
        bb_width_now = float(bb_std[-1]) * 2 / float(bb_mid[-1])
        bb_width_prev = float(bb_std[-5]) * 2 / float(bb_mid[-5]) if bb_mid[-5] > 0 else bb_width_now
        bb_change = (bb_width_now - bb_width_prev) / max(bb_width_prev, 0.001)
    else:
        bb_change = 0.0

    force = float(np.clip(expansion_ratio * 0.6 + bb_change * 0.4, -1, 1))
    return round(force, 4)
