"""
Order Block (OB) Detector

An order block is the last opposing candle before an impulsive directional move.
It marks where institutional orders were placed. When price returns to an OB,
it acts as strong support (bullish OB) or resistance (bearish OB).

Bullish OB: last RED candle before a strong bullish impulse
Bearish OB: last GREEN candle before a strong bearish impulse
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OrderBlock:
    direction: str        # "bullish" | "bearish"
    zone_top: float       # upper edge (max of open/close of the OB candle)
    zone_bottom: float    # lower edge (min of open/close of the OB candle)
    strength: float       # 0-1 based on impulse magnitude
    mitigated: bool       # has price fully traded through the OB?
    candle_index: int
    timestamp: str
    impulse_size: float   # size of the move that followed (in ATR multiples)


def detect_order_blocks(
    df: pd.DataFrame,
    min_impulse_atr: float = 1.5,
    min_impulse_candles: int = 2,
    lookback: int = 50,
) -> list[OrderBlock]:
    """
    Detect order blocks in OHLCV data.

    Args:
        df: OHLCV DataFrame
        min_impulse_atr: minimum impulse size in ATR multiples to qualify
        min_impulse_candles: minimum consecutive candles in impulse direction
        lookback: bars to scan

    Returns:
        List of OrderBlock objects, most recent first
    """
    if len(df) < 5:
        return []

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df.index

    # ATR-14
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    )
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 1.0
    if atr <= 0:
        atr = 1.0

    start = max(1, len(df) - lookback)
    obs: list[OrderBlock] = []

    for i in range(start, len(df) - min_impulse_candles):
        candle_bullish = closes[i] > opens[i]
        candle_bearish = closes[i] < opens[i]

        if not (candle_bullish or candle_bearish):
            continue

        # Check for impulse AFTER this candle
        impulse_dir = None
        impulse_end = i + 1
        total_move = 0.0

        # Count consecutive candles in one direction after candle[i]
        bullish_count = 0
        bearish_count = 0
        for j in range(i + 1, min(i + 8, len(df))):
            if closes[j] > opens[j]:
                bullish_count += 1
                if bearish_count > 0:
                    break
            elif closes[j] < opens[j]:
                bearish_count += 1
                if bullish_count > 0:
                    break
            impulse_end = j

        if bullish_count >= min_impulse_candles:
            impulse_dir = "bullish"
            total_move = max(highs[i + 1:impulse_end + 1]) - lows[i]
        elif bearish_count >= min_impulse_candles:
            impulse_dir = "bearish"
            total_move = highs[i] - min(lows[i + 1:impulse_end + 1])

        if impulse_dir is None:
            continue

        impulse_atr = total_move / atr
        if impulse_atr < min_impulse_atr:
            continue

        # The OB candle must be OPPOSING the impulse
        if impulse_dir == "bullish" and not candle_bearish:
            continue
        if impulse_dir == "bearish" and not candle_bullish:
            continue

        zone_top = max(opens[i], closes[i])
        zone_bottom = min(opens[i], closes[i])
        strength = min(1.0, impulse_atr / 4.0)

        # Check mitigation: has price fully traded through the zone?
        mitigated = False
        for j in range(impulse_end + 1, len(df)):
            if impulse_dir == "bullish" and lows[j] < zone_bottom:
                mitigated = True
                break
            if impulse_dir == "bearish" and highs[j] > zone_top:
                mitigated = True
                break

        obs.append(OrderBlock(
            direction=impulse_dir, zone_top=zone_top, zone_bottom=zone_bottom,
            strength=round(strength, 3), mitigated=mitigated,
            candle_index=i,
            timestamp=str(timestamps[i]) if i < len(timestamps) else "",
            impulse_size=round(impulse_atr, 2),
        ))

    obs.sort(key=lambda o: o.candle_index, reverse=True)
    return obs


def nearest_unmitigated_ob(
    obs: list[OrderBlock], price: float, direction: str = "bullish"
) -> OrderBlock | None:
    """Find nearest unmitigated order block to current price."""
    candidates = [ob for ob in obs if ob.direction == direction and not ob.mitigated]
    if not candidates:
        return None
    candidates.sort(key=lambda ob: abs(ob.zone_top - price) if direction == "bullish"
                    else abs(ob.zone_bottom - price))
    return candidates[0]
