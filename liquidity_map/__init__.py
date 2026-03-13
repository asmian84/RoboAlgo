"""Liquidity Map Engine — identifies areas where stop-loss orders cluster.

Zones detected:
    equal_high_cluster   — repeated equal highs (stop hunts above)
    equal_low_cluster    — repeated equal lows  (stop hunts below)
    previous_day_high    — prior session's high
    previous_day_low     — prior session's low
    swing_high_liquidity — ATR adaptive swing highs (from structure_engine)
    swing_low_liquidity  — ATR adaptive swing lows  (from structure_engine)
    range_high           — high of the recent N-bar consolidation range
    range_low            — low  of the recent N-bar consolidation range
"""

from liquidity_map.liquidity_engine import LiquidityMapEngine

__all__ = ["LiquidityMapEngine"]
