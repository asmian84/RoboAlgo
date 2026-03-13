"""Sweep Detection — detects when price sweeps through a liquidity zone.

A sweep (liquidity grab) occurs when price briefly moves beyond a zone
boundary and then reverses, indicating institutional stop-hunting.

Bottom sweep (zone_type = "low"):
    current_low  < zone_price  AND  close_price > zone_price

Top sweep (zone_type = "high"):
    current_high > zone_price  AND  close_price < zone_price
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def detect_zone_sweep(
    df: pd.DataFrame,
    zone_price: float,
    zone_type: str,
    bar_index: Optional[int] = None,
) -> dict:
    """Detect whether price swept a specific liquidity zone on a given bar.

    Args:
        df:          OHLCV DataFrame with columns: open, high, low, close.
        zone_price:  The price level of the liquidity zone.
        zone_type:   ``"high"`` — zone sits above price (top sweep expected).
                     ``"low"``  — zone sits below price (bottom sweep expected).
        bar_index:   Bar to evaluate.  Defaults to the last bar.

    Returns:
        dict::

            {
                "swept":      bool,   # price moved through zone
                "reclaimed":  bool,   # price closed back on the other side
                "sweep_type": str,    # "top_sweep" | "bottom_sweep" | "none"
                "zone_price": float,
                "zone_type":  str,
            }
    """
    if bar_index is None:
        bar_index = len(df) - 1

    if bar_index >= len(df):
        return _no_sweep(zone_price, zone_type)

    bar = df.iloc[bar_index]

    if zone_type == "high":
        swept     = float(bar["high"])  > zone_price
        reclaimed = float(bar["close"]) < zone_price
        sweep_type = "top_sweep" if (swept and reclaimed) else "none"
    else:
        swept     = float(bar["low"])   < zone_price
        reclaimed = float(bar["close"]) > zone_price
        sweep_type = "bottom_sweep" if (swept and reclaimed) else "none"

    return {
        "swept":      swept,
        "reclaimed":  reclaimed,
        "sweep_type": sweep_type,
        "zone_price": zone_price,
        "zone_type":  zone_type,
    }


def scan_all_zone_sweeps(
    df: pd.DataFrame,
    zones: list[dict],
    lookback: int = 5,
) -> list[dict]:
    """Scan the most recent bars for sweeps of every zone in the map.

    Args:
        df:       OHLCV DataFrame.
        zones:    List of zone dicts — each must contain ``"price"`` and
                  ``"type"`` keys (zone type string from ``ZONE_TYPES``).
        lookback: Number of recent bars to scan.

    Returns:
        List of sweep event dicts::

            {
                "zone":       dict,   # the zone that was swept
                "bar_index":  int,
                "sweep_type": str,    # "top_sweep" | "bottom_sweep"
                "date":       str,
            }
    """
    events: list[dict] = []
    start = max(0, len(df) - lookback)

    for bar_idx in range(start, len(df)):
        for zone in zones:
            zone_price = zone.get("price")
            if zone_price is None:
                continue

            # Determine side from zone type string
            zone_side = "high" if "high" in zone.get("type", "") else "low"
            result = detect_zone_sweep(df, zone_price, zone_side, bar_index=bar_idx)

            if result["sweep_type"] != "none":
                date_val = df.index[bar_idx]
                date_str = (
                    str(date_val.date()) if hasattr(date_val, "date")
                    else str(date_val)
                )
                events.append({
                    "zone":       zone,
                    "bar_index":  bar_idx,
                    "sweep_type": result["sweep_type"],
                    "date":       date_str,
                })

    return events


# ── Internal helpers ───────────────────────────────────────────────────────────

def _no_sweep(zone_price: float, zone_type: str) -> dict:
    return {
        "swept":      False,
        "reclaimed":  False,
        "sweep_type": "none",
        "zone_price": zone_price,
        "zone_type":  zone_type,
    }
