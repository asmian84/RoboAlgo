"""Range Detector — identifies whether price is consolidating in a tight range.

A range (sideways market) is characterised by:
    1. Price oscillating between a support and resistance without breakout.
    2. Low directional slope.
    3. A high/low ratio (range width) that is narrow relative to ATR.

Detection approach
------------------
1. Compute the rolling high and rolling low over ``lookback`` bars.
2. Calculate range width as a fraction of the midpoint price.
3. Classify as RANGE when:
   - width < ``width_threshold``  (price compressed)
   - AND |slope| < ``slope_threshold`` (no directional drift)

Confidence is the inverse of normalised width — tighter compression → higher
confidence that it is a range.

IMPORTANT: ATR is NOT recomputed here.  Only raw OHLCV arithmetic is used.
ATR values are consumed from the caller when needed for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_range(
    df: pd.DataFrame,
    lookback: int = 20,
    width_threshold: float = 0.04,
    slope_threshold: float = 0.001,
) -> dict:
    """Detect whether the market is in a consolidation range.

    Args:
        df:               OHLCV DataFrame with ``high``, ``low``, ``close`` columns.
        lookback:         Number of recent bars to measure the range over.
        width_threshold:  Maximum range width (as fraction of midpoint) to classify
                          as a range.  Default 4%.
        slope_threshold:  Maximum absolute normalised slope to classify as a range.
                          Default 0.001 per bar.

    Returns:
        dict::

            {
                "is_range":         bool,
                "range_confidence": float,  # 0–1
                "range_high":       float,
                "range_low":        float,
                "range_width_pct":  float,  # width as % of midpoint
                "midpoint":         float,
            }
    """
    if len(df) < lookback:
        return _no_range()

    window = df.tail(lookback)
    range_high = float(window["high"].max())
    range_low  = float(window["low"].min())
    midpoint   = (range_high + range_low) / 2.0

    if midpoint == 0:
        return _no_range()

    range_width_pct = (range_high - range_low) / midpoint

    slope = _compute_slope(df, window=lookback)
    is_compressed   = range_width_pct < width_threshold
    is_non_trending = abs(slope) < slope_threshold

    is_range = is_compressed and is_non_trending

    # Confidence rises as width shrinks relative to threshold
    if is_range:
        # 1.0 when width ≈ 0, 0.0 when width == threshold
        width_score  = max(0.0, 1.0 - (range_width_pct / width_threshold))
        slope_score  = max(0.0, 1.0 - (abs(slope) / slope_threshold))
        confidence   = round(0.6 * width_score + 0.4 * slope_score, 4)
    else:
        confidence = 0.0

    return {
        "is_range":         is_range,
        "range_confidence": confidence,
        "range_high":       round(range_high, 4),
        "range_low":        round(range_low,  4),
        "range_width_pct":  round(range_width_pct * 100, 4),   # as percentage
        "midpoint":         round(midpoint, 4),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_slope(df: pd.DataFrame, window: int) -> float:
    """Normalised linear regression slope of close prices (scale-invariant)."""
    if "close" not in df.columns or len(df) < window:
        return 1.0  # unknown → assume trending to avoid false range

    closes = df["close"].values[-window:].astype(float)
    if np.any(np.isnan(closes)) or closes.mean() == 0:
        return 1.0

    x = np.arange(len(closes), dtype=float)
    slope, _ = np.polyfit(x, closes, 1)
    return float(slope / closes.mean())


def _no_range() -> dict:
    return {
        "is_range":         False,
        "range_confidence": 0.0,
        "range_high":       0.0,
        "range_low":        0.0,
        "range_width_pct":  0.0,
        "midpoint":         0.0,
    }
