"""
Market Structure — CHoCH and BOS Detector

BOS (Break of Structure): Trend continuation.
  - Bullish BOS: price breaks above the previous swing high (higher high made)
  - Bearish BOS: price breaks below the previous swing low (lower low made)

CHoCH (Change of Character): Trend reversal — the EARLIEST signal of a top/bottom.
  - Bearish CHoCH: in an uptrend, price breaks below the last higher low
  - Bullish CHoCH: in a downtrend, price breaks above the last lower high

CHoCH is the sniper entry signal — it means smart money has shifted direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StructureShift:
    shift_type: str       # "CHoCH" | "BOS"
    direction: str        # "bullish" | "bearish"
    level: float          # the price level that was broken
    candle_index: int     # bar where the break occurred
    timestamp: str
    confidence: float     # 0-1 based on how decisively the level was broken


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "HH" | "HL" | "LH" | "LL" | "H" | "L"


def _detect_swings(highs: np.ndarray, lows: np.ndarray, order: int = 3) -> list[SwingPoint]:
    """Detect swing highs and lows using a simple N-bar lookback/lookahead."""
    swings: list[SwingPoint] = []
    n = len(highs)

    for i in range(order, n - order):
        # Swing high: highest in window
        if highs[i] == max(highs[i - order:i + order + 1]):
            swings.append(SwingPoint(index=i, price=float(highs[i]), type="H"))

        # Swing low: lowest in window
        if lows[i] == min(lows[i - order:i + order + 1]):
            swings.append(SwingPoint(index=i, price=float(lows[i]), type="L"))

    # Label HH/HL/LH/LL based on sequence
    swing_highs = [s for s in swings if s.type == "H"]
    swing_lows = [s for s in swings if s.type == "L"]

    for i in range(1, len(swing_highs)):
        swing_highs[i].type = "HH" if swing_highs[i].price > swing_highs[i-1].price else "LH"
    for i in range(1, len(swing_lows)):
        swing_lows[i].type = "HL" if swing_lows[i].price > swing_lows[i-1].price else "LL"

    # Merge back and sort by index
    all_swings = swing_highs + swing_lows
    all_swings.sort(key=lambda s: s.index)
    return all_swings


def detect_structure_shifts(
    df: pd.DataFrame,
    swing_order: int = 3,
    lookback: int = 50,
) -> list[StructureShift]:
    """
    Detect CHoCH and BOS events in OHLCV data.

    Returns list of structure shifts, most recent first.

    swing_order is automatically reduced for short datasets so that at least
    a few swing points can be identified.
    """
    if len(df) < 5:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    # Auto-adapt swing order downward until we get at least 4 swings or reach order=1
    effective_order = swing_order
    swings = _detect_swings(highs, lows, order=effective_order)
    while len(swings) < 4 and effective_order > 1:
        effective_order -= 1
        swings = _detect_swings(highs, lows, order=effective_order)
    if len(swings) < 4:
        return []

    shifts: list[StructureShift] = []
    start_idx = max(0, len(df) - lookback)

    # Track the last significant swing high and swing low
    recent_highs = [s for s in swings if s.type in ("HH", "LH", "H") and s.index >= start_idx]
    recent_lows = [s for s in swings if s.type in ("HL", "LL", "L") and s.index >= start_idx]

    for i in range(len(swings) - 1):
        curr = swings[i]
        if curr.index < start_idx:
            continue

        # Find the next few bars after this swing to check for breaks
        for j in range(curr.index + 1, min(curr.index + 15, len(df))):
            # BOS: bullish — close above a swing high (continuation in uptrend)
            if curr.type in ("HH", "H"):
                if closes[j] > curr.price:
                    break_size = (closes[j] - curr.price) / curr.price
                    confidence = min(1.0, break_size * 50)  # 2% break = full confidence
                    shifts.append(StructureShift(
                        shift_type="BOS", direction="bullish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

            # BOS: bearish — close below a swing low (continuation in downtrend)
            if curr.type in ("LL", "L"):
                if closes[j] < curr.price:
                    break_size = (curr.price - closes[j]) / curr.price
                    confidence = min(1.0, break_size * 50)
                    shifts.append(StructureShift(
                        shift_type="BOS", direction="bearish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

            # CHoCH: bearish — in uptrend (HH/HL), price breaks below the last HL
            if curr.type == "HL":
                if closes[j] < curr.price:
                    break_size = (curr.price - closes[j]) / curr.price
                    confidence = min(1.0, break_size * 50)
                    shifts.append(StructureShift(
                        shift_type="CHoCH", direction="bearish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

            # CHoCH: bullish — in downtrend (LH/LL), price breaks above the last LH
            if curr.type == "LH":
                if closes[j] > curr.price:
                    break_size = (closes[j] - curr.price) / curr.price
                    confidence = min(1.0, break_size * 50)
                    shifts.append(StructureShift(
                        shift_type="CHoCH", direction="bullish", level=curr.price,
                        candle_index=j, confidence=round(confidence, 3),
                        timestamp=str(timestamps[j]) if j < len(timestamps) else "",
                    ))
                    break

    shifts.sort(key=lambda s: s.candle_index, reverse=True)
    return shifts


def latest_structure(shifts: list[StructureShift]) -> StructureShift | None:
    """Return the most recent structure shift."""
    return shifts[0] if shifts else None
