"""MACD Pattern detector.

Detects six MACD-based patterns:
  1. MACD Bullish Crossover        — MACD line crosses above signal line
  2. MACD Bearish Crossover        — MACD line crosses below signal line
  3. MACD Zero Line Cross Up       — MACD line crosses above zero (strong bullish)
  4. MACD Zero Line Cross Down     — MACD line crosses below zero (strong bearish)
  5. MACD Histogram Divergence Bull — price lower low, histogram higher low
  6. MACD Histogram Divergence Bear — price higher high, histogram lower high

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

def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, period: int = 14) -> float:
    """Simple ATR using (high - low) over the last *period* bars."""
    return float((high - low).tail(period).mean())


def _swing_highs(series: pd.Series, win: int = 3) -> list[int]:
    indices: list[int] = []
    for i in range(win, len(series) - win):
        window = series.iloc[i - win: i + win + 1]
        if series.iloc[i] == window.max():
            indices.append(i)
    return indices


def _swing_lows(series: pd.Series, win: int = 3) -> list[int]:
    indices: list[int] = []
    for i in range(win, len(series) - win):
        window = series.iloc[i - win: i + win + 1]
        if series.iloc[i] == window.min():
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Individual pattern scanners
# ---------------------------------------------------------------------------

def _detect_crossovers(
    symbol: str,
    df: pd.DataFrame,
    macd: pd.Series,
    signal: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    lookback: int = 10,
) -> list[dict[str, Any]]:
    """Bullish and bearish MACD/signal crossovers."""
    n = len(df)
    results: list[dict[str, Any]] = []
    atr_val = _atr(high, low)

    # Average volume for comparison
    avg_vol = float(volume.tail(20).mean()) if not volume.dropna().empty else 0.0

    start = max(1, n - lookback)
    for i in range(start, n):
        prev_diff = float(macd.iloc[i - 1]) - float(signal.iloc[i - 1])
        curr_diff = float(macd.iloc[i]) - float(signal.iloc[i])
        price     = float(close.iloc[i])
        macd_val  = float(macd.iloc[i])

        if prev_diff <= 0 and curr_diff > 0:
            # Bullish crossover
            conf = 65.0
            if macd_val > 0:
                conf += 10.0  # MACD above zero: momentum confirmation
            if avg_vol > 0 and float(volume.iloc[i]) > avg_vol:
                conf += 5.0   # volume confirmation
            status = "FORMING" if i >= n - 2 else "READY"
            results.append({
                "pattern_name":      "MACD Bullish Crossover",
                "pattern_category":  "indicator",
                "direction":         "bullish",
                "status":            status,
                "confidence":        round(min(conf, 100.0), 2),
                "breakout_level":    round(price + atr_val, 4),
                "invalidation_level": round(price - 1.5 * atr_val, 4),
                "target":            round(price + 2.0 * atr_val, 4),
                "points":            [[i, price]],
            })

        elif prev_diff >= 0 and curr_diff < 0:
            # Bearish crossover
            conf = 65.0
            if macd_val < 0:
                conf += 10.0
            if avg_vol > 0 and float(volume.iloc[i]) > avg_vol:
                conf += 5.0
            status = "FORMING" if i >= n - 2 else "READY"
            results.append({
                "pattern_name":      "MACD Bearish Crossover",
                "pattern_category":  "indicator",
                "direction":         "bearish",
                "status":            status,
                "confidence":        round(min(conf, 100.0), 2),
                "breakout_level":    round(price - atr_val, 4),
                "invalidation_level": round(price + 1.5 * atr_val, 4),
                "target":            round(price - 2.0 * atr_val, 4),
                "points":            [[i, price]],
            })

    return results


def _detect_zero_crosses(
    symbol: str,
    df: pd.DataFrame,
    macd: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 10,
) -> list[dict[str, Any]]:
    """MACD zero-line crossovers (stronger signal than signal-line cross)."""
    n = len(df)
    results: list[dict[str, Any]] = []
    atr_val = _atr(high, low)

    start = max(1, n - lookback)
    for i in range(start, n):
        prev = float(macd.iloc[i - 1])
        curr = float(macd.iloc[i])
        price = float(close.iloc[i])

        if prev <= 0 and curr > 0:
            status = "FORMING" if i >= n - 2 else "READY"
            results.append({
                "pattern_name":      "MACD Zero Line Cross Up",
                "pattern_category":  "indicator",
                "direction":         "bullish",
                "status":            status,
                "confidence":        75.0,
                "breakout_level":    round(price + atr_val, 4),
                "invalidation_level": round(price - 1.5 * atr_val, 4),
                "target":            round(price + 2.0 * atr_val, 4),
                "points":            [[i, price]],
            })

        elif prev >= 0 and curr < 0:
            status = "FORMING" if i >= n - 2 else "READY"
            results.append({
                "pattern_name":      "MACD Zero Line Cross Down",
                "pattern_category":  "indicator",
                "direction":         "bearish",
                "status":            status,
                "confidence":        75.0,
                "breakout_level":    round(price - atr_val, 4),
                "invalidation_level": round(price + 1.5 * atr_val, 4),
                "target":            round(price - 2.0 * atr_val, 4),
                "points":            [[i, price]],
            })

    return results


def _detect_histogram_divergences(
    symbol: str,
    df: pd.DataFrame,
    hist: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 30,
) -> list[dict[str, Any]]:
    """Histogram divergences — price vs MACD histogram."""
    n = len(df)
    results: list[dict[str, Any]] = []
    atr_val = _atr(high, low)

    start = max(0, n - lookback)
    price_slice = close.iloc[start:]
    hist_slice  = hist.iloc[start:]

    price_sh = _swing_highs(price_slice)
    price_sl = _swing_lows(price_slice)
    hist_sh  = _swing_highs(hist_slice)
    hist_sl  = _swing_lows(hist_slice)

    if len(price_sh) >= 2 and len(hist_sh) >= 2:
        p1_local, p2_local = price_sh[-2], price_sh[-1]
        h1_local, h2_local = hist_sh[-2], hist_sh[-1]
        p1_idx = start + p1_local
        p2_idx = start + p2_local
        p1 = float(close.iloc[p1_idx])
        p2 = float(close.iloc[p2_idx])
        h1 = float(hist.iloc[start + h1_local])
        h2 = float(hist.iloc[start + h2_local])
        price_at_p2 = float(close.iloc[p2_idx])

        if p2 > p1 and h2 < h1:
            # Bearish histogram divergence
            results.append({
                "pattern_name":      "MACD Histogram Divergence Bear",
                "pattern_category":  "indicator",
                "direction":         "bearish",
                "status":            "FORMING" if (n - 1 - p2_idx) < 2 else "READY",
                "confidence":        70.0,
                "breakout_level":    round(price_at_p2 - atr_val, 4),
                "invalidation_level": round(price_at_p2 + 1.5 * atr_val, 4),
                "target":            round(price_at_p2 - 2.0 * atr_val, 4),
                "points":            [[p1_idx, p1], [p2_idx, p2]],
            })

    if len(price_sl) >= 2 and len(hist_sl) >= 2:
        p1_local, p2_local = price_sl[-2], price_sl[-1]
        h1_local, h2_local = hist_sl[-2], hist_sl[-1]
        p1_idx = start + p1_local
        p2_idx = start + p2_local
        p1 = float(close.iloc[p1_idx])
        p2 = float(close.iloc[p2_idx])
        h1 = float(hist.iloc[start + h1_local])
        h2 = float(hist.iloc[start + h2_local])
        price_at_p2 = float(close.iloc[p2_idx])

        if p2 < p1 and h2 > h1:
            # Bullish histogram divergence
            results.append({
                "pattern_name":      "MACD Histogram Divergence Bull",
                "pattern_category":  "indicator",
                "direction":         "bullish",
                "status":            "FORMING" if (n - 1 - p2_idx) < 2 else "READY",
                "confidence":        70.0,
                "breakout_level":    round(price_at_p2 + atr_val, 4),
                "invalidation_level": round(price_at_p2 - 1.5 * atr_val, 4),
                "target":            round(price_at_p2 + 2.0 * atr_val, 4),
                "points":            [[p1_idx, p1], [p2_idx, p2]],
            })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect MACD patterns on *df* for *symbol*.

    Parameters
    ----------
    symbol : str
        Ticker symbol (informational — not used in computation).
    df : pd.DataFrame
        OHLCV frame with columns: date, open, high, low, close, volume.
        Must have at least 50 rows; returns [] otherwise.

    Returns
    -------
    list[dict]
        One dict per pattern found, following the pattern-engine service
        contract.  Returns [] on any error or insufficient data.
    """
    try:
        if df is None or len(df) < 50:
            return []

        df = df.sort_values("date").reset_index(drop=True).copy()
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(
            np.zeros(len(df)), index=df.index
        )

        macd, signal, hist = _compute_macd(close)

        results: list[dict[str, Any]] = []
        results.extend(_detect_crossovers(symbol, df, macd, signal, close, high, low, volume))
        results.extend(_detect_zero_crosses(symbol, df, macd, close, high, low))
        results.extend(_detect_histogram_divergences(symbol, df, hist, close, high, low))

        return results

    except Exception:  # noqa: BLE001
        return []
