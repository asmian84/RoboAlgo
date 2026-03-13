"""Trend force: directional strength from moving averages and MACD."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_force(df: pd.DataFrame) -> float:
    """Compute trend force from -1 (strong bearish) to +1 (strong bullish).

    Components:
    - SMA slope direction and magnitude
    - Price position relative to MA50/MA200
    - MACD signal direction
    """
    if df is None or len(df) < 50:
        return 0.0

    close = df["close"].astype(float).values
    n = len(close)

    # MA50 slope
    ma50 = pd.Series(close).rolling(min(50, n)).mean().values
    if n >= 55:
        ma50_slope = (ma50[-1] - ma50[-5]) / max(abs(ma50[-5]), 1e-9)
    else:
        ma50_slope = 0.0

    # Price vs MA50
    price_vs_ma = (close[-1] - ma50[-1]) / max(abs(ma50[-1]), 1e-9) if ma50[-1] > 0 else 0.0

    # MACD signal
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9).mean().values
    macd_hist = float(macd[-1] - macd_signal[-1])
    macd_norm = np.clip(macd_hist / max(abs(close[-1]) * 0.01, 1e-9), -1, 1)

    # Combine
    slope_component = np.clip(ma50_slope * 20, -1, 1) * 0.4
    position_component = np.clip(price_vs_ma * 5, -1, 1) * 0.3
    macd_component = float(macd_norm) * 0.3

    force = float(np.clip(slope_component + position_component + macd_component, -1, 1))
    return round(force, 4)
