"""Trade Setup Engine — converts RoboAlgo signals into complete trade plans.

Each trade plan includes:
    • setup classification  (LIQUIDITY_REVERSAL | TREND_PULLBACK | BREAKOUT_EXPANSION)
    • trade type            (DAY_TRADE | SWING_TRADE | INVESTMENT)
    • entry price
    • stop loss (placed beyond key liquidity levels)
    • profit targets        (T1=1.5R, T2=3R, T3=next liquidity zone)
    • risk-reward ratio
    • confidence score
    • timeframe alignment   (bias / setup / entry TF)
    • position sizing       (fixed-fractional from account risk parameters)
"""

from trade_setup.setup_engine import (
    TradeSetupEngine,
    TRADE_TYPE_DAY,
    TRADE_TYPE_SWING,
    TRADE_TYPE_INVEST,
)
from trade_setup.entry_logic import (
    SETUP_LIQUIDITY_REVERSAL,
    SETUP_TREND_PULLBACK,
    SETUP_BREAKOUT_EXPANSION,
    DIRECTION_LONG,
    DIRECTION_SHORT,
)

__all__ = [
    "TradeSetupEngine",
    "TRADE_TYPE_DAY",
    "TRADE_TYPE_SWING",
    "TRADE_TYPE_INVEST",
    "SETUP_LIQUIDITY_REVERSAL",
    "SETUP_TREND_PULLBACK",
    "SETUP_BREAKOUT_EXPANSION",
    "DIRECTION_LONG",
    "DIRECTION_SHORT",
]
