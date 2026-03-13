"""Trend Detector — classifies price action as trending up, trending down, or neutral.

Trend classification is built on two complementary signals:

1. **Swing structure** (primary) — consumed from structure_engine.swing_detector.
   A bullish trend requires a sequence of Higher Highs (HH) and Higher Lows (HL).
   A bearish trend requires Lower Highs (LH) and Lower Lows (LL).

2. **Slope confirmation** (secondary) — a simple linear regression slope over the
   most recent ``slope_window`` bars of close prices, normalised by the mean close
   so that it is scale-invariant.

Only *swing structure* gates the trend classification.  Slope adds confidence
but cannot flip a neutral classification to trending on its own.

IMPORTANT: ATR calculation is NOT performed here.  This module consumes swing
pivot data and computes only trend structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Minimum consecutive swing pivots required to declare a trend
_MIN_PIVOTS = 3


def detect_trend(
    df: pd.DataFrame,
    swing_highs: list[dict],
    swing_lows: list[dict],
    slope_window: int = 30,
) -> dict:
    """Classify the current market trend using swing structure and price slope.

    Args:
        df:           OHLCV DataFrame (close column required for slope).
        swing_highs:  Swing high dicts from structure_engine — each must
                      contain ``"index"`` and ``"price"`` keys.
        swing_lows:   Swing low dicts with the same schema.
        slope_window: Number of recent bars used for slope regression.

    Returns:
        dict::

            {
                "trend":          "UP" | "DOWN" | "NEUTRAL",
                "trend_strength": float,   # 0–1 normalised confidence
                "hh_count":       int,     # consecutive higher highs
                "hl_count":       int,     # consecutive higher lows
                "lh_count":       int,     # consecutive lower highs
                "ll_count":       int,     # consecutive lower lows
                "slope":          float,   # normalised linear regression slope
            }
    """
    slope = _compute_slope(df, window=slope_window)

    # Sort pivots by bar index (ascending)
    sh_sorted = sorted(swing_highs, key=lambda x: x.get("index", 0))
    sl_sorted = sorted(swing_lows,  key=lambda x: x.get("index", 0))

    hh_count = _count_consecutive_higher(sh_sorted)
    hl_count = _count_consecutive_higher(sl_sorted)
    lh_count = _count_consecutive_lower(sh_sorted)
    ll_count = _count_consecutive_lower(sl_sorted)

    # Primary classification from swing structure
    bullish_structure = hh_count >= _MIN_PIVOTS and hl_count >= _MIN_PIVOTS
    bearish_structure = lh_count >= _MIN_PIVOTS and ll_count >= _MIN_PIVOTS

    if bullish_structure and slope > 0:
        trend = "UP"
        # Strength: average of normalised pivot count and positive slope magnitude
        pivot_score  = min((hh_count + hl_count) / 12, 1.0)   # caps at 6+6 pivots
        slope_score  = min(abs(slope) * 50, 1.0)               # 2% daily = 1.0
        trend_strength = round(0.7 * pivot_score + 0.3 * slope_score, 4)
    elif bearish_structure and slope < 0:
        trend = "DOWN"
        pivot_score  = min((lh_count + ll_count) / 12, 1.0)
        slope_score  = min(abs(slope) * 50, 1.0)
        trend_strength = round(0.7 * pivot_score + 0.3 * slope_score, 4)
    elif bullish_structure:
        trend = "UP"
        trend_strength = round(min((hh_count + hl_count) / 12, 1.0) * 0.6, 4)
    elif bearish_structure:
        trend = "DOWN"
        trend_strength = round(min((lh_count + ll_count) / 12, 1.0) * 0.6, 4)
    else:
        trend = "NEUTRAL"
        trend_strength = 0.0

    return {
        "trend":          trend,
        "trend_strength": trend_strength,
        "hh_count":       hh_count,
        "hl_count":       hl_count,
        "lh_count":       lh_count,
        "ll_count":       ll_count,
        "slope":          round(slope, 6),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _count_consecutive_higher(pivots: list[dict]) -> int:
    """Count consecutive Higher pivots (each price > previous) from the right."""
    prices = [p["price"] for p in pivots]
    count = 0
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] > prices[i - 1]:
            count += 1
        else:
            break
    return count


def _count_consecutive_lower(pivots: list[dict]) -> int:
    """Count consecutive Lower pivots (each price < previous) from the right."""
    prices = [p["price"] for p in pivots]
    count = 0
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] < prices[i - 1]:
            count += 1
        else:
            break
    return count


def _compute_slope(df: pd.DataFrame, window: int) -> float:
    """Compute the normalised linear regression slope of close prices.

    Slope is normalised by the mean close price so the result is scale-invariant
    (i.e. comparable across different price levels).

    Returns 0.0 if there is insufficient data or close is missing.
    """
    if "close" not in df.columns or len(df) < window:
        return 0.0

    closes = df["close"].values[-window:].astype(float)
    if np.any(np.isnan(closes)) or closes.mean() == 0:
        return 0.0

    x = np.arange(len(closes), dtype=float)
    slope, _ = np.polyfit(x, closes, 1)

    # Normalise by mean close to get a percentage-per-bar rate
    return float(slope / closes.mean())
