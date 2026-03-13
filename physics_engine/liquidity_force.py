"""Liquidity force: directional pull from nearby liquidity pools."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_liquidity_force(df: pd.DataFrame) -> float:
    """Compute liquidity force from -1 (pulled down) to +1 (pulled up).

    Based on inverse distance to key liquidity levels (recent highs/lows,
    volume-weighted price zones).
    """
    if df is None or len(df) < 20:
        return 0.0

    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    vol = df["volume"].fillna(0).astype(float).values

    current = float(close[-1])
    if current <= 0:
        return 0.0

    # Key levels: recent highs and lows
    high_20 = float(np.max(high[-20:]))
    low_20 = float(np.min(low[-20:]))
    high_5 = float(np.max(high[-5:]))
    low_5 = float(np.min(low[-5:]))

    # VWAP approximation
    if vol[-20:].sum() > 0:
        vwap = float(np.average(close[-20:], weights=vol[-20:]))
    else:
        vwap = float(np.mean(close[-20:]))

    # Compute pull forces (inverse distance, capped)
    def _pull(level: float) -> float:
        dist = (level - current) / max(abs(current), 1e-9)
        if abs(dist) < 0.001:
            return 0.0
        # Stronger pull when closer (inverse relationship)
        strength = 1.0 / (abs(dist) * 100 + 1.0)
        return strength if dist > 0 else -strength

    pulls = [
        _pull(high_20) * 0.3,
        _pull(low_20) * 0.3,
        _pull(high_5) * 0.15,
        _pull(low_5) * 0.15,
        _pull(vwap) * 0.1,
    ]

    force = float(np.clip(sum(pulls), -1, 1))
    return round(force, 4)
