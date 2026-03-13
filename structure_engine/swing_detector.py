"""Swing high/low detection using a zigzag-style minimum move filter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_adaptive_minimum_move(
    price_data: pd.DataFrame,
    atr_period: int = 14,
    atr_multiplier: float = 0.5,
    floor: float = 0.01,
) -> float:
    """Compute ATR-adaptive minimum_move for swing detection.

    Formula: max(atr_multiplier * ATR_14 / close, floor)
    - Low-vol stocks (AAPL ~1.5% daily) → ~0.75% threshold
    - High-vol stocks (TQQQ ~4% daily) → ~2% threshold
    - Floor of 1% prevents degenerate pivots in ultra-smooth data
    """
    if price_data is None or len(price_data) < atr_period + 1:
        return 0.03  # fallback to legacy default

    high = price_data["high"].astype(float) if "high" in price_data.columns else price_data["close"].astype(float)
    low = price_data["low"].astype(float) if "low" in price_data.columns else price_data["close"].astype(float)
    close = price_data["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = float(tr.tail(atr_period).mean())
    current_close = float(close.iloc[-1])

    if current_close <= 0 or np.isnan(atr):
        return 0.03

    return max(atr_multiplier * atr / current_close, floor)


def detect_swings(price_data: pd.DataFrame, minimum_move: float = 0.03) -> dict[str, list[dict[str, Any]]]:
    """Return swing highs/lows from OHLCV data using zigzag threshold."""
    if price_data is None or price_data.empty:
        return {"swing_highs": [], "swing_lows": []}

    df = price_data.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    df = df.sort_values("date").reset_index(drop=True)
    if "close" not in df.columns or len(df) < 3:
        return {"swing_highs": [], "swing_lows": []}

    close = df["close"].astype(float).tolist()
    highs = df.get("high", df["close"]).astype(float).tolist()
    lows = df.get("low", df["close"]).astype(float).tolist()
    dates = df["date"].tolist()

    pivots: list[dict[str, Any]] = []
    last_idx = 0
    last_price = close[0]
    trend = 0  # 1 up, -1 down, 0 unknown
    extreme_idx = 0
    extreme_price = close[0]

    for i in range(1, len(close)):
        px = close[i]
        move = (px - last_price) / max(abs(last_price), 1e-9)

        if trend == 0:
            if abs(move) >= minimum_move:
                trend = 1 if move > 0 else -1
                extreme_idx = i
                extreme_price = highs[i] if trend > 0 else lows[i]
            continue

        if trend > 0:
            if highs[i] >= extreme_price:
                extreme_price = highs[i]
                extreme_idx = i
            drawdown = (lows[i] - extreme_price) / max(abs(extreme_price), 1e-9)
            if drawdown <= -minimum_move:
                pivots.append(
                    {
                        "index": extreme_idx,
                        "date": dates[extreme_idx],
                        "price": float(extreme_price),
                        "type": "H",
                    }
                )
                last_idx = extreme_idx
                last_price = extreme_price
                trend = -1
                extreme_idx = i
                extreme_price = lows[i]
        else:
            if lows[i] <= extreme_price:
                extreme_price = lows[i]
                extreme_idx = i
            runup = (highs[i] - extreme_price) / max(abs(extreme_price), 1e-9)
            if runup >= minimum_move:
                pivots.append(
                    {
                        "index": extreme_idx,
                        "date": dates[extreme_idx],
                        "price": float(extreme_price),
                        "type": "L",
                    }
                )
                last_idx = extreme_idx
                last_price = extreme_price
                trend = 1
                extreme_idx = i
                extreme_price = highs[i]

    if trend > 0 and extreme_idx != last_idx:
        pivots.append({"index": extreme_idx, "date": dates[extreme_idx], "price": float(extreme_price), "type": "H"})
    elif trend < 0 and extreme_idx != last_idx:
        pivots.append({"index": extreme_idx, "date": dates[extreme_idx], "price": float(extreme_price), "type": "L"})

    swing_highs = [p for p in pivots if p["type"] == "H"]
    swing_lows = [p for p in pivots if p["type"] == "L"]
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}

