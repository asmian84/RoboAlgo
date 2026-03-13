"""Gann Square-of-9 price level calculator.

The Square-of-9 arranges prices in a spiral pattern where geometrically
related prices share angular positions. Key levels are found by rotating
the square root of price by 45°, 90°, 180°, 270°, 360° increments.

Formula: level = (sqrt(price) ± n * increment)^2
where increment = 2.0 (standard Gann increment = one full revolution)
and n = rotation count.
"""

from __future__ import annotations

import math


def square_of_9_levels(
    price: float,
    n_levels: int = 4,
    increment: float = 2.0,
) -> dict:
    """Compute Square-of-9 support and resistance levels.

    Args:
        price: Current price.
        n_levels: Number of levels above and below.
        increment: Gann increment (2.0 = one full revolution of the square).

    Returns:
        Dict with 'support' and 'resistance' lists of price levels.
    """
    if price <= 0:
        return {"support": [], "resistance": []}

    sqrt_price = math.sqrt(price)

    resistance: list[dict] = []
    support: list[dict] = []

    for i in range(1, n_levels + 1):
        # Full rotations
        res_val = (sqrt_price + i * increment) ** 2
        sup_val = (sqrt_price - i * increment) ** 2
        resistance.append({
            "level": round(res_val, 4),
            "rotation": i,
            "distance_pct": round((res_val - price) / price * 100, 2),
        })
        if sup_val > 0:
            support.append({
                "level": round(sup_val, 4),
                "rotation": i,
                "distance_pct": round((sup_val - price) / price * 100, 2),
            })

        # Cardinal cross (90° increments = 0.5 in sqrt space)
        for angle, label in [(0.25, "45°"), (0.5, "90°"), (1.0, "180°"), (1.5, "270°")]:
            r = (sqrt_price + i * increment * angle / increment) ** 2
            if r > price and len(resistance) < n_levels * 5:
                resistance.append({
                    "level": round(r, 4),
                    "rotation": i,
                    "angle": label,
                    "distance_pct": round((r - price) / price * 100, 2),
                })

    resistance.sort(key=lambda x: x["level"])
    support.sort(key=lambda x: x["level"], reverse=True)

    return {"support": support[:n_levels], "resistance": resistance[:n_levels]}


def sq9_nearest_levels(price: float) -> tuple[float, float]:
    """Return (nearest_support, nearest_resistance) from Square-of-9."""
    levels = square_of_9_levels(price, n_levels=2)
    sup = levels["support"][0]["level"] if levels["support"] else price * 0.95
    res = levels["resistance"][0]["level"] if levels["resistance"] else price * 1.05
    return sup, res
