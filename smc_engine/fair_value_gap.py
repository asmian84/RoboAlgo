"""
smc_engine/fair_value_gap.py
────────────────────────────────────────────────────────────────────────────────
Fair Value Gap (FVG) detector for Smart Money Concepts.

A Fair Value Gap is a three-candle imbalance zone created when price moves so
aggressively that it leaves a gap between the wicks of the candles surrounding
the impulse candle.  Price tends to return to fill these zones — they act as
magnets for precision entries.

Detection rules (candle[i] is the impulse candle):
  Bullish FVG : candle[i-1].high  < candle[i+1].low   → gap above prior candle
  Bearish FVG : candle[i-1].low   > candle[i+1].high  → gap below prior candle

Strength is the gap size expressed as a fraction of the 14-period ATR, capped
at 1.0.  Only gaps ≥ min_gap_atr × ATR are returned.

An FVG is considered "filled" once price has traded back inside the gap zone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class FVG:
    """Represents a single Fair Value Gap zone."""

    direction: str          # "bullish" | "bearish"
    top: float              # upper edge of the gap
    bottom: float           # lower edge of the gap
    midpoint: float         # centre of the gap — optimal entry reference
    strength: float         # 0.0 – 1.0  (gap / ATR, capped at 1.0)
    filled: bool            # True once price has re-entered the zone
    candle_index: int       # positional index of the impulse candle in the df
    timestamp: str          # ISO-format string of the impulse candle's datetime


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return a Series of Average True Range values (Wilder's method)."""
    high = df["high"]
    low  = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Use a simple rolling mean for the first window, then Wilder's EWM
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def _is_filled(fvg: "FVG", df: pd.DataFrame) -> bool:
    """
    Return True if any candle *after the three-candle FVG pattern* has traded
    back inside the gap zone.

    The pattern occupies indices [i-1, i, i+1].  The i+1 candle's edge defines
    one boundary of the zone, so we start checking from i+2 to avoid a
    false-positive from the defining candle itself.

    Fill conditions:
      Bullish FVG: a subsequent candle's low dips into (or below) the gap zone,
                   i.e. low < fvg.top  (price returned to fill from above)
      Bearish FVG: a subsequent candle's high rallies into (or above) the gap,
                   i.e. high > fvg.bottom  (price returned to fill from below)
    """
    # candle_index is the impulse (i); the pattern ends at i+1, so start at i+2
    start = fvg.candle_index + 2
    subsequent = df.iloc[start:]
    if subsequent.empty:
        return False

    if fvg.direction == "bullish":
        # Price must dip below the top edge to enter the gap from above
        touched = subsequent["low"] < fvg.top
    else:
        # Price must rally above the bottom edge to enter the gap from below
        touched = subsequent["high"] > fvg.bottom

    return bool(touched.any())


# ── Public API ────────────────────────────────────────────────────────────────

def detect_fvgs(
    df: pd.DataFrame,
    min_gap_atr: float = 0.3,
    lookback: int = 50,
    check_filled: bool = True,
) -> list[FVG]:
    """
    Detect Fair Value Gaps in a price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with lowercase columns (open, high, low, close, volume).
        Index must be datetime-like.
    min_gap_atr : float
        Minimum gap size as a multiple of ATR to qualify as an FVG.
        Default 0.3 (30 % of ATR).
    lookback : int
        Number of recent candles to scan.  Older candles are ignored.
    check_filled : bool
        When True, each FVG is marked as filled if subsequent price action
        has traded back inside the zone.

    Returns
    -------
    list[FVG]
        Detected FVGs, most-recent impulse candle first.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    if len(df) < 3:
        return []

    atr_series = _compute_atr(df)

    # Only scan the last `lookback` candles; we still need i-1 and i+1 so
    # the scannable range inside that window is [start+1 … end-1].
    scan_end   = len(df) - 1           # last valid impulse (needs i+1)
    scan_start = max(1, len(df) - lookback)  # needs i-1

    fvgs: list[FVG] = []

    for i in range(scan_start, scan_end):
        prev  = df.iloc[i - 1]
        curr  = df.iloc[i]      # impulse candle (not used directly in gap calc)
        nxt   = df.iloc[i + 1]
        atr   = atr_series.iloc[i]

        if pd.isna(atr) or atr <= 0:
            continue

        # ── Bullish FVG ───────────────────────────────────────────────────────
        if prev["high"] < nxt["low"]:
            gap_size = nxt["low"] - prev["high"]
            if gap_size >= min_gap_atr * atr:
                top      = nxt["low"]
                bottom   = prev["high"]
                midpoint = (top + bottom) / 2.0
                strength = min(gap_size / atr, 1.0)
                timestamp = str(df.index[i])

                fvg = FVG(
                    direction="bullish",
                    top=top,
                    bottom=bottom,
                    midpoint=midpoint,
                    strength=strength,
                    filled=False,
                    candle_index=i,
                    timestamp=timestamp,
                )
                if check_filled:
                    fvg.filled = _is_filled(fvg, df)
                fvgs.append(fvg)

        # ── Bearish FVG ───────────────────────────────────────────────────────
        elif prev["low"] > nxt["high"]:
            gap_size = prev["low"] - nxt["high"]
            if gap_size >= min_gap_atr * atr:
                top      = prev["low"]
                bottom   = nxt["high"]
                midpoint = (top + bottom) / 2.0
                strength = min(gap_size / atr, 1.0)
                timestamp = str(df.index[i])

                fvg = FVG(
                    direction="bearish",
                    top=top,
                    bottom=bottom,
                    midpoint=midpoint,
                    strength=strength,
                    filled=False,
                    candle_index=i,
                    timestamp=timestamp,
                )
                if check_filled:
                    fvg.filled = _is_filled(fvg, df)
                fvgs.append(fvg)

    # Most-recent impulse candle first
    fvgs.sort(key=lambda f: f.candle_index, reverse=True)
    return fvgs


def nearest_unfilled_fvg(
    fvgs: list[FVG],
    price: float,
    direction: str = "bullish",
) -> Optional[FVG]:
    """
    Return the unfilled FVG of the given direction that is closest to *price*.

    Proximity is measured from *price* to the FVG midpoint.  Returns None if
    no qualifying FVG exists.

    Parameters
    ----------
    fvgs : list[FVG]
        Output of :func:`detect_fvgs`.
    price : float
        Current market price.
    direction : str
        "bullish" or "bearish".

    Returns
    -------
    FVG | None
    """
    candidates = [
        f for f in fvgs
        if not f.filled and f.direction == direction
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda f: abs(f.midpoint - price))
