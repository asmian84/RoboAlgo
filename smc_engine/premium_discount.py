"""
smc_engine/premium_discount.py
──────────────────────────────────────────────────────────────────────────────
Premium/Discount Zone Classifier + Optimal Trade Entry (OTE)

The range between the most recent significant swing high and swing low defines
the "dealing range." Price above the 50% mark is PREMIUM (sell zone), below is
DISCOUNT (buy zone).

OTE (Optimal Trade Entry) is the 62-79% Fibonacci retracement zone — the sweet
spot where smart money enters on pullbacks.

Rules:
  - Only buy in DISCOUNT zone (below equilibrium)
  - Only sell in PREMIUM zone (above equilibrium)
  - Best entries are at OTE level within the correct zone
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ZoneAnalysis:
    swing_high: float         # top of dealing range
    swing_low: float          # bottom of dealing range
    equilibrium: float        # 50% level (midpoint)
    current_price: float
    zone: str                 # "PREMIUM" | "DISCOUNT" | "EQUILIBRIUM"
    zone_pct: float           # where price sits in the range (0=low, 100=high)
    ote_top: float            # OTE zone upper bound (62% retracement from swing high)
    ote_bottom: float         # OTE zone lower bound (79% retracement from swing high)
    in_ote: bool              # is price within the OTE zone?
    discount_levels: list[float]  # key fib levels in discount zone
    premium_levels: list[float]   # key fib levels in premium zone


def analyze_zone(
    df: pd.DataFrame,
    swing_lookback: int = 50,
    swing_order: int = 5,
) -> ZoneAnalysis:
    """
    Classify current price as premium/discount within the dealing range.

    Uses the most recent significant swing high and swing low to define the range.
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    price = float(closes[-1])
    n = len(df)

    # Find significant swing high and low in lookback window
    window = min(swing_lookback, n)
    recent_highs = highs[-window:]
    recent_lows = lows[-window:]

    swing_high = float(np.max(recent_highs))
    swing_low = float(np.min(recent_lows))
    dealing_range = swing_high - swing_low

    if dealing_range <= 0:
        return ZoneAnalysis(
            swing_high=swing_high, swing_low=swing_low,
            equilibrium=price, current_price=price,
            zone="EQUILIBRIUM", zone_pct=50.0,
            ote_top=price, ote_bottom=price, in_ote=False,
            discount_levels=[], premium_levels=[],
        )

    equilibrium = swing_low + dealing_range * 0.50
    zone_pct = (price - swing_low) / dealing_range * 100

    if zone_pct > 55:
        zone = "PREMIUM"
    elif zone_pct < 45:
        zone = "DISCOUNT"
    else:
        zone = "EQUILIBRIUM"

    # OTE zone: 62-79% retracement from swing high (= 21-38% from swing low)
    # For a bullish OTE (buying the dip):
    ote_top = swing_high - dealing_range * 0.618     # 61.8% retracement
    ote_bottom = swing_high - dealing_range * 0.786   # 78.6% retracement
    in_ote = ote_bottom <= price <= ote_top

    # Key fib levels
    fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
    all_levels = [swing_low + dealing_range * f for f in fib_levels]
    discount_levels = [l for l in all_levels if l < equilibrium]
    premium_levels = [l for l in all_levels if l > equilibrium]

    return ZoneAnalysis(
        swing_high=round(swing_high, 4), swing_low=round(swing_low, 4),
        equilibrium=round(equilibrium, 4), current_price=round(price, 4),
        zone=zone, zone_pct=round(zone_pct, 1),
        ote_top=round(ote_top, 4), ote_bottom=round(ote_bottom, 4),
        in_ote=in_ote,
        discount_levels=[round(l, 4) for l in discount_levels],
        premium_levels=[round(l, 4) for l in premium_levels],
    )
