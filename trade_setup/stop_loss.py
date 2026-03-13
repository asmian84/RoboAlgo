"""Stop Loss Logic — places stops beyond key liquidity levels.

Stop placement rules per setup
-------------------------------
LIQUIDITY_REVERSAL
    Stop beyond the swept liquidity level (the wick that triggered the sweep).
    LONG:  stop = swept_low  - buffer
    SHORT: stop = swept_high + buffer

TREND_PULLBACK
    Stop beyond the pullback swing extremum.
    LONG:  stop = pullback_swing_low  - buffer
    SHORT: stop = pullback_swing_high + buffer

BREAKOUT_EXPANSION
    Stop inside the broken range (below range_high for longs, above range_low
    for shorts) — the breakout is invalid if price re-enters the range.
    LONG:  stop = range_low  - buffer
    SHORT: stop = range_high + buffer

Buffer = fraction of the entry price used to push the stop slightly beyond
the structural level (default 0.2%).  This avoids triggering on minor noise.

IMPORTANT: ATR is NOT recomputed here.  When the caller wants ATR-based
buffer, it should pass the ATR value directly as ``atr_buffer``.
"""

from __future__ import annotations

import pandas as pd

from trade_setup.entry_logic import (
    DIRECTION_LONG,
    DIRECTION_SHORT,
    SETUP_LIQUIDITY_REVERSAL,
    SETUP_TREND_PULLBACK,
    SETUP_BREAKOUT_EXPANSION,
)

# Default buffer: 0.2% of entry price, or 0.25× ATR when ATR is provided
_DEFAULT_BUFFER_PCT = 0.002
_ATR_BUFFER_MULT    = 0.25


def compute_stop_loss(
    entry: dict,
    df: pd.DataFrame,
    setup_type: str,
    *,
    regime: dict | None = None,
    nearest_zone: dict | None = None,
    atr: float | None = None,
) -> dict:
    """Calculate stop loss price for a given entry.

    Args:
        entry:          Output of :func:`entry_logic.compute_entry`.
        df:             OHLCV DataFrame.
        setup_type:     One of the three setup type constants.
        regime:         Market Regime Engine output (for BREAKOUT_EXPANSION).
        nearest_zone:   Closest liquidity zone from LiquidityMapEngine.
        atr:            Current ATR value.  When supplied, the buffer is
                        ``atr × _ATR_BUFFER_MULT``; otherwise ``entry × 0.2%``.

    Returns:
        dict::

            {
                "stop_loss":    float,
                "stop_type":    str,    # human-readable label
                "stop_distance":float,  # |entry - stop_loss|
            }
    """
    entry_price = entry["entry"]
    direction   = entry["direction"]
    bar_index   = entry["bar_index"]

    # Compute buffer: prefer ATR-based, fallback to percentage
    if atr and atr > 0:
        buffer = round(atr * _ATR_BUFFER_MULT, 4)
    else:
        buffer = round(entry_price * _DEFAULT_BUFFER_PCT, 4)

    if setup_type == SETUP_LIQUIDITY_REVERSAL:
        stop, label = _stop_liquidity_reversal(
            entry_price, direction, buffer,
            nearest_zone,
        )
    elif setup_type == SETUP_TREND_PULLBACK:
        stop, label = _stop_trend_pullback(
            entry_price, direction, buffer, df, bar_index,
        )
    elif setup_type == SETUP_BREAKOUT_EXPANSION:
        stop, label = _stop_breakout_expansion(
            entry_price, direction, buffer, regime, nearest_zone,
        )
    else:
        # Fallback: flat buffer stop
        stop  = entry_price - buffer if direction == DIRECTION_LONG else entry_price + buffer
        label = "BUFFER"

    stop_distance = abs(entry_price - stop)

    return {
        "stop_loss":     round(stop, 4),
        "stop_type":     label,
        "stop_distance": round(stop_distance, 4),
    }


# ── Setup-specific stop rules ──────────────────────────────────────────────────

def _stop_liquidity_reversal(
    entry: float,
    direction: str,
    buffer: float,
    nearest_zone: dict | None,
) -> tuple[float, str]:
    """Stop beyond the swept liquidity level."""
    # Use nearest zone price
    if nearest_zone:
        zone_price = float(nearest_zone.get("price", entry))
        if direction == DIRECTION_LONG:
            return round(zone_price - buffer, 4), "ZONE_LOW_STOP"
        else:
            return round(zone_price + buffer, 4), "ZONE_HIGH_STOP"

    # Last resort: flat buffer
    if direction == DIRECTION_LONG:
        return round(entry - buffer * 4, 4), "BUFFER_STOP"
    else:
        return round(entry + buffer * 4, 4), "BUFFER_STOP"


def _stop_trend_pullback(
    entry: float,
    direction: str,
    buffer: float,
    df: pd.DataFrame,
    bar_index: int,
) -> tuple[float, str]:
    """Stop beyond the pullback swing extremum."""
    lookback = min(10, bar_index)
    start    = max(0, bar_index - lookback)
    window   = df.iloc[start:bar_index + 1]

    if window.empty:
        stop = entry - buffer * 3 if direction == DIRECTION_LONG else entry + buffer * 3
        return round(stop, 4), "BUFFER_STOP"

    if direction == DIRECTION_LONG:
        swing_low = float(window["low"].min())
        return round(swing_low - buffer, 4), "PULLBACK_LOW_STOP"
    else:
        swing_high = float(window["high"].max())
        return round(swing_high + buffer, 4), "PULLBACK_HIGH_STOP"


def _stop_breakout_expansion(
    entry: float,
    direction: str,
    buffer: float,
    regime: dict | None,
    nearest_zone: dict | None,
) -> tuple[float, str]:
    """Stop inside the broken range boundary."""
    if regime:
        range_high = float(regime.get("range_high", 0.0))
        range_low  = float(regime.get("range_low",  0.0))

        if direction == DIRECTION_LONG and range_low > 0:
            return round(range_low - buffer, 4), "RANGE_LOW_STOP"
        if direction == DIRECTION_SHORT and range_high > 0:
            return round(range_high + buffer, 4), "RANGE_HIGH_STOP"

    # Fallback to zone-based stop
    if nearest_zone:
        zone_price = float(nearest_zone.get("price", entry))
        if direction == DIRECTION_LONG:
            return round(zone_price - buffer, 4), "ZONE_LOW_STOP"
        else:
            return round(zone_price + buffer, 4), "ZONE_HIGH_STOP"

    if direction == DIRECTION_LONG:
        return round(entry - buffer * 3, 4), "BUFFER_STOP"
    else:
        return round(entry + buffer * 3, 4), "BUFFER_STOP"
