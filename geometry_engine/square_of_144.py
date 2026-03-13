"""Gann Square-of-144 price level calculator.

Similar to Square-of-9 but uses 144 as the base cycle (Fibonacci number).
Levels are found at 1/8, 1/4, 1/3, 3/8, 1/2, 5/8, 2/3, 3/4, 7/8 divisions
of each 144-unit range that contains the current price.
"""

from __future__ import annotations

import math

# Division fractions of a 144-unit range
SQ144_DIVISIONS = [
    (1/8, "1/8"),
    (1/4, "2/8"),
    (1/3, "1/3"),
    (3/8, "3/8"),
    (1/2, "1/2"),
    (5/8, "5/8"),
    (2/3, "2/3"),
    (3/4, "6/8"),
    (7/8, "7/8"),
]


def square_of_144_levels(price: float, range_size: float = 144.0) -> dict:
    """Compute Square-of-144 support and resistance levels.

    Finds the 144-unit range containing the price, then computes
    division levels within that range and adjacent ranges.
    """
    if price <= 0:
        return {"support": [], "resistance": [], "range_low": 0, "range_high": 0}

    # Find which 144-unit range contains the price
    range_num = int(price / range_size)
    range_low = range_num * range_size
    range_high = range_low + range_size

    support: list[dict] = []
    resistance: list[dict] = []

    # Generate levels for current range and one range above/below
    for r_offset in [-1, 0, 1]:
        r_low = (range_num + r_offset) * range_size
        r_high = r_low + range_size
        for frac, label in SQ144_DIVISIONS:
            level = r_low + (r_high - r_low) * frac
            if level <= 0:
                continue
            entry = {
                "level": round(level, 4),
                "label": label,
                "range_offset": r_offset,
                "distance_pct": round((level - price) / price * 100, 2),
            }
            if level < price:
                support.append(entry)
            elif level > price:
                resistance.append(entry)

    support.sort(key=lambda x: x["level"], reverse=True)
    resistance.sort(key=lambda x: x["level"])

    return {
        "support": support[:6],
        "resistance": resistance[:6],
        "range_low": round(range_low, 4),
        "range_high": round(range_high, 4),
    }


def sq144_nearest_levels(price: float) -> tuple[float, float]:
    """Return (nearest_support, nearest_resistance) from Square-of-144."""
    levels = square_of_144_levels(price)
    sup = levels["support"][0]["level"] if levels["support"] else price * 0.95
    res = levels["resistance"][0]["level"] if levels["resistance"] else price * 1.05
    return sup, res
