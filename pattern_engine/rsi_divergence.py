"""RSI Divergence detector.

Detects four divergence types:
  - Regular Bullish:  price lower low, RSI higher low  → bullish reversal
  - Regular Bearish:  price higher high, RSI lower high → bearish reversal
  - Hidden Bullish:   price higher low, RSI lower low   → bullish continuation
  - Hidden Bearish:   price lower high, RSI higher high → bearish continuation

Service contract
----------------
detect(symbol, df) -> list[dict]

Each dict exposes:
  pattern_name, pattern_category, status, direction, confidence,
  breakout_level, invalidation_level, target, points
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def _swing_highs(series: pd.Series, win: int = 3) -> list[int]:
    """Return bar indices that are local swing highs (centre of a 2*win+1 window)."""
    indices: list[int] = []
    for i in range(win, len(series) - win):
        window = series.iloc[i - win: i + win + 1]
        if series.iloc[i] == window.max():
            indices.append(i)
    return indices


def _swing_lows(series: pd.Series, win: int = 3) -> list[int]:
    """Return bar indices that are local swing lows."""
    indices: list[int] = []
    for i in range(win, len(series) - win):
        window = series.iloc[i - win: i + win + 1]
        if series.iloc[i] == window.min():
            indices.append(i)
    return indices


def _confidence(rsi1: float, rsi2: float, base: float = 60.0, scale: float = 50.0) -> float:
    """Confidence grows with the absolute RSI distance between the two pivots."""
    raw = base + abs(rsi2 - rsi1) / scale * (85.0 - base)
    return float(np.clip(raw, base, 85.0))


# ---------------------------------------------------------------------------
# Core divergence scanner
# ---------------------------------------------------------------------------

def _scan(
    symbol: str,
    df: pd.DataFrame,
    lookback: int = 50,
    recency: int = 10,
    recent_confirm: int = 2,
) -> list[dict[str, Any]]:
    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    n     = len(df)

    if n < 30:
        return []

    rsi = _compute_rsi(close)

    # Restrict search window
    start = max(0, n - lookback)
    price_slice = close.iloc[start:]
    rsi_slice   = rsi.iloc[start:]

    sh_local = _swing_highs(price_slice)
    sl_local = _swing_lows(price_slice)

    rsh_local = _swing_highs(rsi_slice)
    rsl_local = _swing_lows(rsi_slice)

    # Convert local indices back to global bar indices
    sh  = [start + i for i in sh_local]
    sl  = [start + i for i in sl_local]
    rsh = [start + i for i in rsh_local]
    rsl = [start + i for i in rsl_local]

    # Need at least 2 of each pivot type to compare
    if len(sh) < 2 or len(sl) < 2 or len(rsh) < 2 or len(rsl) < 2:
        return []

    results: list[dict[str, Any]] = []
    recent_close = float(close.iloc[-1])
    recent_high  = float(high.tail(20).max())
    recent_low   = float(low.tail(20).min())

    def _status(second_pivot_idx: int) -> str:
        bars_ago = (n - 1) - second_pivot_idx
        return "FORMING" if bars_ago < recent_confirm else "READY"

    # -----------------------------------------------------------------------
    # Regular Bullish: last 2 price lows → lower low; last 2 RSI lows → higher low
    # -----------------------------------------------------------------------
    p1_idx, p2_idx = sl[-2], sl[-1]
    r1_idx, r2_idx = rsl[-2], rsl[-1]
    bars_ago = (n - 1) - p2_idx
    if bars_ago <= recency:
        p1, p2 = float(close.iloc[p1_idx]), float(close.iloc[p2_idx])
        r1, r2 = float(rsi.iloc[r1_idx]), float(rsi.iloc[r2_idx])
        if p2 < p1 and r2 > r1:
            conf = _confidence(r1, r2)
            results.append({
                "pattern_name":      "Regular Bullish Divergence",
                "pattern_category":  "indicator",
                "direction":         "bullish",
                "status":            _status(p2_idx),
                "confidence":        round(conf, 2),
                "breakout_level":    round(recent_high * 1.01, 4),
                "invalidation_level": round(recent_low * 0.98, 4),
                "target":            round(recent_high * 1.05, 4),
                "points":            [[p1_idx, p1], [p2_idx, p2]],
            })

    # -----------------------------------------------------------------------
    # Regular Bearish: last 2 price highs → higher high; last 2 RSI highs → lower high
    # -----------------------------------------------------------------------
    p1_idx, p2_idx = sh[-2], sh[-1]
    r1_idx, r2_idx = rsh[-2], rsh[-1]
    bars_ago = (n - 1) - p2_idx
    if bars_ago <= recency:
        p1, p2 = float(close.iloc[p1_idx]), float(close.iloc[p2_idx])
        r1, r2 = float(rsi.iloc[r1_idx]), float(rsi.iloc[r2_idx])
        if p2 > p1 and r2 < r1:
            conf = _confidence(r1, r2)
            results.append({
                "pattern_name":      "Regular Bearish Divergence",
                "pattern_category":  "indicator",
                "direction":         "bearish",
                "status":            _status(p2_idx),
                "confidence":        round(conf, 2),
                "breakout_level":    round(recent_low * 0.99, 4),
                "invalidation_level": round(recent_high * 1.02, 4),
                "target":            round(recent_low * 0.95, 4),
                "points":            [[p1_idx, p1], [p2_idx, p2]],
            })

    # -----------------------------------------------------------------------
    # Hidden Bullish: price higher low, RSI lower low → bullish continuation
    # -----------------------------------------------------------------------
    p1_idx, p2_idx = sl[-2], sl[-1]
    r1_idx, r2_idx = rsl[-2], rsl[-1]
    bars_ago = (n - 1) - p2_idx
    if bars_ago <= recency:
        p1, p2 = float(close.iloc[p1_idx]), float(close.iloc[p2_idx])
        r1, r2 = float(rsi.iloc[r1_idx]), float(rsi.iloc[r2_idx])
        if p2 > p1 and r2 < r1:
            conf = _confidence(r1, r2)
            results.append({
                "pattern_name":      "Hidden Bullish Divergence",
                "pattern_category":  "indicator",
                "direction":         "bullish",
                "status":            _status(p2_idx),
                "confidence":        round(conf, 2),
                "breakout_level":    round(recent_high * 1.01, 4),
                "invalidation_level": round(recent_low * 0.98, 4),
                "target":            round(recent_high * 1.05, 4),
                "points":            [[p1_idx, p1], [p2_idx, p2]],
            })

    # -----------------------------------------------------------------------
    # Hidden Bearish: price lower high, RSI higher high → bearish continuation
    # -----------------------------------------------------------------------
    p1_idx, p2_idx = sh[-2], sh[-1]
    r1_idx, r2_idx = rsh[-2], rsh[-1]
    bars_ago = (n - 1) - p2_idx
    if bars_ago <= recency:
        p1, p2 = float(close.iloc[p1_idx]), float(close.iloc[p2_idx])
        r1, r2 = float(rsi.iloc[r1_idx]), float(rsi.iloc[r2_idx])
        if p2 < p1 and r2 > r1:
            conf = _confidence(r1, r2)
            results.append({
                "pattern_name":      "Hidden Bearish Divergence",
                "pattern_category":  "indicator",
                "direction":         "bearish",
                "status":            _status(p2_idx),
                "confidence":        round(conf, 2),
                "breakout_level":    round(recent_low * 0.99, 4),
                "invalidation_level": round(recent_high * 1.02, 4),
                "target":            round(recent_low * 0.95, 4),
                "points":            [[p1_idx, p1], [p2_idx, p2]],
            })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect RSI divergences on *df* for *symbol*.

    Parameters
    ----------
    symbol : str
        Ticker symbol (informational — not used in computation).
    df : pd.DataFrame
        OHLCV frame with columns: date, open, high, low, close, volume.

    Returns
    -------
    list[dict]
        One dict per divergence found, following the pattern-engine service
        contract.  Returns [] on any error or if the frame is too short.
    """
    try:
        if df is None or len(df) < 30:
            return []
        return _scan(symbol, df)
    except Exception:  # noqa: BLE001
        return []
