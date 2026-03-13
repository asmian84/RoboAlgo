"""Entry Logic — determines entry price and direction for each setup type.

Entry rules per setup
---------------------
LIQUIDITY_REVERSAL
    Entry on the close of the sweep candle (confirmed reversal close).
    Direction is opposite to the sweep direction:
        bottom sweep → LONG entry  (above swept low)
        top    sweep → SHORT entry (below swept high)

TREND_PULLBACK
    Entry at the structure break price — the level where price breaks
    above the recent swing high (long) or below the recent swing low
    (short), confirming the pullback has ended.

BREAKOUT_EXPANSION
    Entry at the breakout candle's close beyond the range boundary.
    Long breakout → close > range_high
    Short breakout → close < range_low

IMPORTANT: This module does NOT recompute price levels.  All input prices
are consumed from the upstream engine outputs supplied by setup_engine.py.
"""

from __future__ import annotations

import pandas as pd


# ── Setup type constants ───────────────────────────────────────────────────────

SETUP_LIQUIDITY_REVERSAL  = "LIQUIDITY_REVERSAL"
SETUP_TREND_PULLBACK      = "TREND_PULLBACK"
SETUP_BREAKOUT_EXPANSION  = "BREAKOUT_EXPANSION"

ALL_SETUPS = [SETUP_LIQUIDITY_REVERSAL, SETUP_TREND_PULLBACK, SETUP_BREAKOUT_EXPANSION]

# Direction constants
DIRECTION_LONG  = "LONG"
DIRECTION_SHORT = "SHORT"


def compute_entry(
    df: pd.DataFrame,
    setup_type: str,
    bar_index: int,
    *,
    regime: dict | None = None,
    zone: dict | None = None,
) -> dict:
    """Determine the entry price and direction for a given setup.

    Args:
        df:             OHLCV DataFrame.
        setup_type:     One of ``ALL_SETUPS``.
        bar_index:      Bar at which the setup fires.
        regime:         Market Regime Engine output dict (used for TREND_PULLBACK
                        and BREAKOUT_EXPANSION).
        zone:           Nearest liquidity zone dict from LiquidityMapEngine
                        (used for LIQUIDITY_REVERSAL and BREAKOUT_EXPANSION).

    Returns:
        dict::

            {
                "entry":     float,   # entry price
                "direction": str,     # "LONG" | "SHORT"
                "entry_type":str,     # human-readable entry label
                "bar_index": int,
            }
    """
    bar = df.iloc[bar_index]
    close = float(bar["close"])

    if setup_type == SETUP_LIQUIDITY_REVERSAL:
        return _entry_liquidity_reversal(close, bar_index)

    if setup_type == SETUP_TREND_PULLBACK:
        return _entry_trend_pullback(df, bar_index, close, regime)

    if setup_type == SETUP_BREAKOUT_EXPANSION:
        return _entry_breakout_expansion(close, bar_index, regime, zone)

    # Fallback — should not be reached
    return {
        "entry":      close,
        "direction":  DIRECTION_LONG,
        "entry_type": "CLOSE",
        "bar_index":  bar_index,
    }


# ── Setup-specific entry rules ─────────────────────────────────────────────────

def _entry_liquidity_reversal(
    close: float,
    bar_index: int,
) -> dict:
    """Entry on the sweep candle's close in the reversal direction."""
    return {
        "entry":      round(close, 4),
        "direction":  DIRECTION_LONG,
        "entry_type": "SWEEP_CLOSE",
        "bar_index":  bar_index,
    }


def _entry_trend_pullback(
    df: pd.DataFrame,
    bar_index: int,
    close: float,
    regime: dict | None,
) -> dict:
    """Entry at the pullback structure break price.

    For a bullish pullback:  entry = close once it reclaims prior swing high.
    For a bearish pullback:  entry = close once it breaks prior swing low.
    """
    trend = (regime or {}).get("trend", "NEUTRAL")
    direction = DIRECTION_LONG if trend == "UP" else DIRECTION_SHORT

    # Locate the recent structure level the pullback is retesting
    lookback = min(10, bar_index)
    start = max(0, bar_index - lookback)
    window = df.iloc[start:bar_index]

    if not window.empty:
        if direction == DIRECTION_LONG:
            # Entry at the most recent swing high that close has reclaimed
            structure_price = float(window["high"].max())
        else:
            structure_price = float(window["low"].min())
    else:
        structure_price = close

    entry = round(close, 4)   # confirmed on close through structure

    return {
        "entry":             entry,
        "direction":         direction,
        "entry_type":        "STRUCTURE_BREAK",
        "bar_index":         bar_index,
        "structure_level":   round(structure_price, 4),
    }


def _entry_breakout_expansion(
    close: float,
    bar_index: int,
    regime: dict | None,
    zone: dict | None,
) -> dict:
    """Entry on the breakout candle close beyond the range boundary."""
    if regime:
        range_high = regime.get("range_high", 0.0)
        range_low  = regime.get("range_low",  0.0)
        if range_high and range_low:
            direction = DIRECTION_LONG if close > range_high else DIRECTION_SHORT
        else:
            direction = DIRECTION_LONG
    elif zone:
        side = zone.get("side", "high")
        direction = DIRECTION_LONG if side == "high" else DIRECTION_SHORT
    else:
        direction = DIRECTION_LONG

    return {
        "entry":      round(close, 4),
        "direction":  direction,
        "entry_type": "BREAKOUT_CLOSE",
        "bar_index":  bar_index,
    }
