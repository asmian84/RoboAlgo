"""
tests/test_smc_engine/test_fvg.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for the Fair Value Gap detector (smc_engine.fair_value_gap).
"""
from __future__ import annotations

import pandas as pd
import pytest

from smc_engine.fair_value_gap import FVG, detect_fvgs, nearest_unfilled_fvg


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    # Ensure volume column present
    if "volume" not in df.columns:
        df["volume"] = 1_000_000
    return df[["open", "high", "low", "close", "volume"]]


def _flat_candle(price: float) -> dict:
    return {"open": price, "high": price, "low": price, "close": price, "volume": 1_000_000}


# ── Test 1: detect bullish FVG ────────────────────────────────────────────────

def test_detects_bullish_fvg():
    """
    Three-candle pattern where candle[i-1].high < candle[i+1].low by a wide
    margin creates a textbook bullish FVG.

    Layout (prices):
      candle 0 (prev) : high = 100
      candle 1 (impulse): strong up-move
      candle 2 (next) : low  = 110  → gap = 100–110 (10 pts)

    We pad with 20 flat candles before the pattern so the ATR is well-defined,
    and 5 flat candles after so we have room for the lookback to include i+1.
    The flat section sits at 100, giving an ATR close to 0.  We therefore
    build a series where each candle has a small (1-pt) range to give a
    realistic ATR, then insert the large-gap pattern.
    """
    # 20 "warm-up" candles with a 1-pt range around 100
    rows = [{"open": 99.5, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1_000_000}] * 20

    # The three-candle pattern
    rows.append({"open": 99.0,  "high": 100.0, "low": 99.0,  "close": 100.0, "volume": 1_000_000})  # i-1, high=100
    rows.append({"open": 100.0, "high": 115.0, "low": 100.0, "close": 114.0, "volume": 2_000_000})  # i   (impulse)
    rows.append({"open": 114.0, "high": 116.0, "low": 112.0, "close": 115.0, "volume": 1_500_000})  # i+1, low=112

    # 5 candles afterwards that do NOT fill the gap (stay above 112)
    rows += [{"open": 115.0, "high": 116.0, "low": 114.0, "close": 115.0, "volume": 1_000_000}] * 5

    df = _make_df(rows)
    fvgs = detect_fvgs(df, min_gap_atr=0.3, lookback=50, check_filled=True)

    bullish = [f for f in fvgs if f.direction == "bullish"]
    assert len(bullish) >= 1, "Should detect at least one bullish FVG"

    fvg = bullish[0]
    assert fvg.bottom == pytest.approx(100.0), "Bottom should be prev candle's high"
    assert fvg.top    == pytest.approx(112.0), "Top should be next candle's low"
    assert fvg.midpoint == pytest.approx(106.0)
    assert 0.0 < fvg.strength <= 1.0
    assert fvg.filled is False


# ── Test 2: detect bearish FVG ────────────────────────────────────────────────

def test_detects_bearish_fvg():
    """
    Three-candle pattern where candle[i-1].low > candle[i+1].high creates a
    textbook bearish FVG.

    Layout:
      candle i-1 : low  = 100
      candle i   : strong down-move (impulse)
      candle i+1 : high = 90   → gap = 90–100 (10 pts)
    """
    rows = [{"open": 100.5, "high": 101.0, "low": 99.5, "close": 100.0, "volume": 1_000_000}] * 20

    rows.append({"open": 101.0, "high": 101.5, "low": 100.0, "close": 100.5, "volume": 1_000_000})  # i-1, low=100
    rows.append({"open": 100.5, "high": 100.5, "low": 86.0,  "close": 87.0,  "volume": 2_000_000})  # i   (impulse)
    rows.append({"open": 87.0,  "high": 89.0,  "low": 85.0,  "close": 86.0,  "volume": 1_500_000})  # i+1, high=89

    # 5 candles after that do NOT fill the gap (stay below 100)
    rows += [{"open": 86.0, "high": 87.5, "low": 85.0, "close": 86.5, "volume": 1_000_000}] * 5

    df = _make_df(rows)
    fvgs = detect_fvgs(df, min_gap_atr=0.3, lookback=50, check_filled=True)

    bearish = [f for f in fvgs if f.direction == "bearish"]
    assert len(bearish) >= 1, "Should detect at least one bearish FVG"

    fvg = bearish[0]
    assert fvg.bottom == pytest.approx(89.0), "Bottom should be next candle's high"
    assert fvg.top    == pytest.approx(100.0), "Top should be prev candle's low"
    assert fvg.midpoint == pytest.approx(94.5)
    assert 0.0 < fvg.strength <= 1.0
    assert fvg.filled is False


# ── Test 3: flat data returns no FVGs ─────────────────────────────────────────

def test_returns_empty_on_flat_data():
    """
    When all candles have the same OHLC, there is no gap anywhere, so detect_fvgs
    should return an empty list.
    """
    rows = [_flat_candle(100.0)] * 30
    df = _make_df(rows)
    fvgs = detect_fvgs(df, min_gap_atr=0.3, lookback=50)
    assert fvgs == [], f"Expected empty list, got {fvgs}"


# ── Test 4: FVG has all required fields ───────────────────────────────────────

def test_fvg_has_required_fields():
    """
    Every FVG returned must contain all fields defined in the dataclass.
    """
    rows = [{"open": 99.5, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1_000_000}] * 20
    rows.append({"open": 99.0, "high": 100.0, "low": 99.0, "close": 100.0, "volume": 1_000_000})
    rows.append({"open": 100.0, "high": 115.0, "low": 100.0, "close": 114.0, "volume": 2_000_000})
    rows.append({"open": 114.0, "high": 116.0, "low": 112.0, "close": 115.0, "volume": 1_500_000})
    rows += [{"open": 115.0, "high": 116.0, "low": 114.0, "close": 115.0, "volume": 1_000_000}] * 5

    df = _make_df(rows)
    fvgs = detect_fvgs(df, min_gap_atr=0.3, lookback=50)

    assert len(fvgs) >= 1, "Need at least one FVG to inspect fields"
    fvg = fvgs[0]

    # Check all required dataclass fields exist and have the right type
    assert isinstance(fvg.direction,    str),   "direction must be str"
    assert isinstance(fvg.top,          float), "top must be float"
    assert isinstance(fvg.bottom,       float), "bottom must be float"
    assert isinstance(fvg.midpoint,     float), "midpoint must be float"
    assert isinstance(fvg.strength,     float), "strength must be float"
    assert isinstance(fvg.filled,       bool),  "filled must be bool"
    assert isinstance(fvg.candle_index, int),   "candle_index must be int"
    assert isinstance(fvg.timestamp,    str),   "timestamp must be str"

    # Direction must be one of the two valid values
    assert fvg.direction in {"bullish", "bearish"}
    # Strength must be in the valid range
    assert 0.0 <= fvg.strength <= 1.0
    # Geometric sanity
    assert fvg.bottom <= fvg.midpoint <= fvg.top
    assert fvg.top > fvg.bottom


# ── Test 5: nearest_unfilled_fvg ─────────────────────────────────────────────

def test_nearest_unfilled_fvg():
    """
    Given a list of FVGs with varying midpoints, nearest_unfilled_fvg should
    return the closest unfilled one to the supplied price.
    """
    def _fvg(direction, bottom, top, filled=False):
        return FVG(
            direction=direction,
            top=top,
            bottom=bottom,
            midpoint=(top + bottom) / 2.0,
            strength=0.5,
            filled=filled,
            candle_index=0,
            timestamp="2024-01-01 00:00:00",
        )

    fvgs = [
        _fvg("bullish", 90.0,  92.0),        # midpoint = 91, unfilled
        _fvg("bullish", 98.0,  102.0),        # midpoint = 100, unfilled
        _fvg("bullish", 108.0, 112.0),        # midpoint = 110, unfilled
        _fvg("bullish", 115.0, 119.0, filled=True),  # midpoint = 117, FILLED
        _fvg("bearish", 105.0, 107.0),        # bearish — should be excluded
    ]

    price = 103.0

    result = nearest_unfilled_fvg(fvgs, price, direction="bullish")
    assert result is not None, "Should find a nearest unfilled FVG"
    # midpoint 100 is distance 3, midpoint 110 is distance 7, midpoint 91 is 12
    assert result.midpoint == pytest.approx(100.0), (
        f"Expected midpoint 100 (closest unfilled bullish), got {result.midpoint}"
    )
    assert result.filled is False

    # When direction is bearish, only bearish FVGs should be considered
    result_bear = nearest_unfilled_fvg(fvgs, price, direction="bearish")
    assert result_bear is not None
    assert result_bear.direction == "bearish"

    # When all are filled, return None
    all_filled = [_fvg("bullish", 90.0, 92.0, filled=True)]
    assert nearest_unfilled_fvg(all_filled, price, direction="bullish") is None

    # Empty list → None
    assert nearest_unfilled_fvg([], price) is None
