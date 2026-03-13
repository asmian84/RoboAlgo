"""Pattern Validation Engine.

Computes confluence confidence from:
- pattern strength
- structure alignment
- liquidity alignment
- momentum confirmation
- volume confirmation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(x, lo, hi))


def _volume_confirmation(df: pd.DataFrame) -> float:
    if "volume" not in df.columns or len(df) < 20:
        return 50.0
    vol = df["volume"].fillna(0.0).astype(float).tail(21)
    baseline = float(vol.iloc[:-1].mean())
    if baseline <= 0:
        return 50.0
    ratio = float(vol.iloc[-1]) / baseline
    return _clamp(45 + ratio * 25)


def _momentum_confirmation(df: pd.DataFrame) -> float:
    if len(df) < 25:
        return 50.0
    close = df["close"].astype(float)
    ret5 = float(close.iloc[-1] / max(close.iloc[-6], 1e-9) - 1.0) if len(close) >= 6 else 0.0
    ret20 = float(close.iloc[-1] / max(close.iloc[-21], 1e-9) - 1.0) if len(close) >= 21 else 0.0
    return _clamp(50 + ret5 * 400 + ret20 * 150)


def _liquidity_alignment(df: pd.DataFrame, breakout: float | None) -> float:
    if breakout is None or len(df) < 30:
        return 50.0
    highs = df["high"].astype(float).tail(40)
    lows = df["low"].astype(float).tail(40)
    hi = float(highs.max())
    lo = float(lows.min())
    width = max(hi - lo, 1e-9)
    distance = min(abs(hi - breakout), abs(breakout - lo)) / width
    return _clamp(100 - distance * 180)


def _structure_alignment(points: list[list[float]] | None) -> float:
    if not points or len(points) < 2:
        return 45.0
    # Points may carry integer bar-indices OR date strings (post-normalization).
    # If the first element is a string, fall back to sequential integers so the
    # regression still measures price linearity regardless of x-axis units.
    try:
        x = np.array([p[0] for p in points], dtype=float)
    except (ValueError, TypeError):
        x = np.arange(len(points), dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    if len(x) < 3:
        return 60.0
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    err = float(np.mean(np.abs(y - fit) / np.maximum(np.abs(y), 1e-6)))
    return _clamp(85 - err * 220)


def compute_confluence(pattern: dict[str, Any], price_data: pd.DataFrame) -> dict[str, float]:
    """Return confluence score and components."""
    pattern_strength = _clamp(float(pattern.get("confidence", 0.0) or 0.0))
    structure = _structure_alignment(pattern.get("points"))
    liquidity = _liquidity_alignment(price_data, pattern.get("breakout_level"))
    momentum = _momentum_confirmation(price_data)
    volume = _volume_confirmation(price_data)

    score = _clamp(
        0.30 * pattern_strength
        + 0.25 * structure
        + 0.20 * liquidity
        + 0.15 * momentum
        + 0.10 * volume
    )
    return {
        "confluence_score": round(score, 2),
        "pattern_strength": round(pattern_strength, 2),
        "structure_alignment": round(structure, 2),
        "liquidity_alignment": round(liquidity, 2),
        "momentum_confirmation": round(momentum, 2),
        "volume_confirmation": round(volume, 2),
    }

