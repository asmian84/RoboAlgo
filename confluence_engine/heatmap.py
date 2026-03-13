"""Confluence Heatmap Generator.

Generates a price-time heatmap showing where multiple analysis engines
agree on significant price levels. Used for chart overlay visualization.

The heatmap is a 2D grid:
  - X axis: time (bars from current)
  - Y axis: price (levels within ±ATR*N of current price)
  - Value:  confluence intensity (0-1) — how many engines agree at that point
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from cycle_engine.cycle_projection import project_cycle
from indicator_engine.technical import atr_scalar as _atr_scalar
from geometry_engine.square_of_9 import square_of_9_levels
from geometry_engine.square_of_144 import square_of_144_levels
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

logger = logging.getLogger("confluence_engine.heatmap")


def generate_heatmap(
    df: pd.DataFrame,
    symbol: str = "",
    n_price_bins: int = 40,
    n_time_bins: int = 20,
    price_range_atr: float = 5.0,
) -> dict[str, Any]:
    """Generate a confluence heatmap for chart overlay.

    Args:
        df: Price DataFrame with date, open, high, low, close, volume.
        symbol: Ticker symbol.
        n_price_bins: Number of vertical (price) bins.
        n_time_bins: Number of horizontal (time) bins into the future.
        price_range_atr: ATR multiples above/below current price for the grid.

    Returns:
        Dict with heatmap data:
          - price_axis: list of price levels (y axis)
          - time_axis: list of bar offsets (x axis, 0 = now)
          - intensity: 2D list [price_idx][time_idx] with 0-1 values
          - levels: list of individual contributor levels for annotation
    """
    result: dict[str, Any] = {
        "symbol": symbol,
        "price_axis": [],
        "time_axis": [],
        "intensity": [],
        "levels": [],
    }

    if df is None or len(df) < 60:
        return result

    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float).values
    current_price = float(close[-1])

    # Compute ATR for price range — delegate to canonical indicator_engine.technical
    high = df["high"].astype(float).values
    low  = df["low"].astype(float).values
    atr  = _atr_scalar(high, low, close, period=14)

    if atr <= 0 or current_price <= 0:
        return result

    # ── Build axes ─────────────────────────────────────────────────────────
    price_low = current_price - price_range_atr * atr
    price_high = current_price + price_range_atr * atr
    price_axis = np.linspace(price_low, price_high, n_price_bins).tolist()
    time_axis = list(range(n_time_bins))

    result["price_axis"] = [round(p, 4) for p in price_axis]
    result["time_axis"] = time_axis

    # Initialize intensity grid
    intensity = np.zeros((n_price_bins, n_time_bins))

    # ── Collect contributor levels ─────────────────────────────────────────
    levels: list[dict] = []

    # 1. Swing S/R levels (horizontal lines → constant across time)
    try:
        adaptive_mm = compute_adaptive_minimum_move(df)
        swings = detect_swings(df, minimum_move=adaptive_mm)
        for sl in swings["swing_lows"][-5:]:
            price = float(sl["price"])
            if price_low <= price <= price_high:
                levels.append({"price": price, "source": "swing_low", "weight": 0.8})
        for sh in swings["swing_highs"][-5:]:
            price = float(sh["price"])
            if price_low <= price <= price_high:
                levels.append({"price": price, "source": "swing_high", "weight": 0.8})
    except Exception:
        pass

    # 2. Square-of-9 levels
    try:
        sq9 = square_of_9_levels(current_price, n_levels=3)
        for lv in sq9.get("support", []) + sq9.get("resistance", []):
            price = lv["level"]
            if price_low <= price <= price_high:
                levels.append({"price": price, "source": "sq9", "weight": 0.6})
    except Exception:
        pass

    # 3. Square-of-144 levels
    try:
        sq144 = square_of_144_levels(current_price)
        for lv in sq144.get("support", []) + sq144.get("resistance", []):
            price = lv["level"]
            if price_low <= price <= price_high:
                levels.append({"price": price, "source": "sq144", "weight": 0.5})
    except Exception:
        pass

    # 4. Cycle projections (time-dependent)
    try:
        cycle_data = project_cycle(df)
        peak_price = cycle_data.get("next_peak_price")
        trough_price = cycle_data.get("next_trough_price")
        dominant_cycle = cycle_data.get("dominant_cycle_length", 20)

        if peak_price and price_low <= peak_price <= price_high:
            # Peak appears at a specific time offset
            bars_to_peak = ((0.25 - cycle_data.get("cycle_phase", 0)) % 1.0) * dominant_cycle
            levels.append({"price": peak_price, "source": "cycle_peak", "weight": 0.7, "time_offset": bars_to_peak})

        if trough_price and price_low <= trough_price <= price_high:
            bars_to_trough = ((0.75 - cycle_data.get("cycle_phase", 0)) % 1.0) * dominant_cycle
            levels.append({"price": trough_price, "source": "cycle_trough", "weight": 0.7, "time_offset": bars_to_trough})
    except Exception:
        pass

    # ── Paint levels onto the heatmap grid ────────────────────────────────
    price_step = (price_high - price_low) / max(n_price_bins - 1, 1)

    for lv in levels:
        price = lv["price"]
        weight = lv["weight"]

        # Find the nearest price bin
        price_idx = int(round((price - price_low) / max(price_step, 1e-9)))
        price_idx = max(0, min(n_price_bins - 1, price_idx))

        time_offset = lv.get("time_offset")

        if time_offset is not None:
            # Time-dependent level (cycle peaks/troughs)
            time_idx = int(round(time_offset))
            time_idx = max(0, min(n_time_bins - 1, time_idx))
            # Gaussian spread around the point
            for dp in range(-2, 3):
                for dt in range(-2, 3):
                    pi = price_idx + dp
                    ti = time_idx + dt
                    if 0 <= pi < n_price_bins and 0 <= ti < n_time_bins:
                        dist = (dp ** 2 + dt ** 2) ** 0.5
                        falloff = max(0, 1.0 - dist / 3.0)
                        intensity[pi][ti] += weight * falloff
        else:
            # Horizontal level (constant across all time bins)
            for dp in range(-1, 2):
                pi = price_idx + dp
                if 0 <= pi < n_price_bins:
                    falloff = 1.0 - abs(dp) * 0.3
                    for ti in range(n_time_bins):
                        intensity[pi][ti] += weight * falloff * 0.5

    # Normalize intensity to 0-1
    max_intensity = float(intensity.max())
    if max_intensity > 0:
        intensity /= max_intensity

    result["intensity"] = [[round(float(v), 3) for v in row] for row in intensity.tolist()]
    result["levels"] = levels

    return result
