"""Market Regime Engine — classifies the current market environment.

Supported regimes:
    TREND_UP             Confirmed bullish structure (HH + HL swing sequence).
    TREND_DOWN           Confirmed bearish structure (LH + LL swing sequence).
    RANGE                Sideways price action with no directional bias.
    VOLATILITY_EXPANSION ATR significantly above its rolling mean.
    ACCUMULATION         Range with bullish liquidity sweep activity.
    DISTRIBUTION         Range with bearish liquidity sweep activity.
"""

from market_regime.regime_engine import MarketRegimeEngine
from market_regime.regime_classifier import (
    ALL_REGIMES,
    REGIME_TREND_UP,
    REGIME_TREND_DOWN,
    REGIME_RANGE,
    REGIME_VOL_EXP,
    REGIME_ACCUM,
    REGIME_DISTRIB,
)

__all__ = [
    "MarketRegimeEngine",
    "ALL_REGIMES",
    "REGIME_TREND_UP",
    "REGIME_TREND_DOWN",
    "REGIME_RANGE",
    "REGIME_VOL_EXP",
    "REGIME_ACCUM",
    "REGIME_DISTRIB",
]
