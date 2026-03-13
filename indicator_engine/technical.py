"""
RoboAlgo — Shared Vectorised Indicator Primitives
==================================================
Fast numpy implementations of core technical indicators used across the
signal and pattern engines.  Pandas-based engines should use
``indicator_engine.calculator.IndicatorCalculator`` for DB-backed work.
These functions operate directly on ``np.ndarray`` for hot-path scanning.

Exported functions
------------------
    rolling_mean(arr, window)            → ndarray
    rolling_std(arr, window)             → ndarray
    true_range(high, low, close)         → ndarray
    atr(high, low, close, period=14)     → ndarray   (Wilder EMA smoothing)
    bollinger(close, period=20, std=2.0) → (mid, upper, lower, bw)
    keltner(high, low, close, period=20, mult=1.5) → (kc_upper, kc_lower)
    percentile_rank(series, window)      → ndarray   (rolling pct rank 0-1)
    momentum_oscillator(close, period=12) → ndarray  (close − EMA)
    ema(arr, period)                     → ndarray   (exponential MA)
"""
from __future__ import annotations

import numpy as np


# ── Rolling helpers ───────────────────────────────────────────────────────────

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """O(n) cumsum rolling mean — no pandas dependency."""
    out = np.full(arr.shape, np.nan)
    cs  = np.cumsum(arr)
    out[window - 1:] = (cs[window - 1:] - np.concatenate([[0], cs[:-window]])) / window
    return out


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling population std (ddof=0)."""
    out = np.full(arr.shape, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = arr[i - window + 1 : i + 1].std(ddof=0)
    return out


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average — alpha = 2/(period+1)."""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    alpha = 2.0 / (period + 1)
    result[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


# ── ATR family ────────────────────────────────────────────────────────────────

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range (Wilder definition)."""
    n      = len(high)
    tr     = np.empty(n)
    tr[0]  = high[0] - low[0]
    pc     = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - pc),
                                   np.abs(low[1:] - pc)))
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range — Wilder EMA smoothing (alpha = 1/period)."""
    tr   = true_range(high, low, close)
    out  = np.full_like(tr, np.nan)
    if len(tr) < period:
        return out
    out[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
    return out


def atr_scalar(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14, floor_pct: float = 0.005) -> float:
    """
    Return the most-recent ATR value as a Python float.
    ``floor_pct`` prevents zero ATR (e.g. 0.005 = 0.5% of last close).
    """
    a = atr(high, low, close, period)
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        return float(close[-1]) * floor_pct
    return max(float(valid[-1]), float(close[-1]) * floor_pct)


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def bollinger(
    close: np.ndarray,
    period: int = 20,
    std_mult: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.
    Returns (mid, upper, lower, bandwidth) — all ndarray, same length as close.
    """
    mid   = rolling_mean(close, period)
    std   = rolling_std(close, period)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw    = np.where(mid > 1e-9, (upper - lower) / mid, np.nan)
    return mid, upper, lower, bw


# ── Keltner Channels ──────────────────────────────────────────────────────────

def keltner(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    mult: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keltner Channels (ATR-based).
    Returns (kc_upper, kc_lower).
    """
    mid     = rolling_mean(close, period)
    atr_arr = atr(high, low, close, period)
    return mid + mult * atr_arr, mid - mult * atr_arr


# ── Statistical helpers ───────────────────────────────────────────────────────

def percentile_rank(series: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling percentile rank (0.0–1.0): fraction of the look-back window
    whose value is ≤ the current bar.  NaN for first (window-1) bars.
    """
    n   = len(series)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_vals = series[i - window + 1 : i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < 2:
            continue
        out[i] = float(np.sum(valid <= series[i]) / len(valid))
    return out


# ── Momentum ──────────────────────────────────────────────────────────────────

def momentum_oscillator(close: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Simple momentum: close − EMA(close, period).
    Proxy for the TTM Squeeze momentum histogram.
    """
    return close - ema(close, period)
