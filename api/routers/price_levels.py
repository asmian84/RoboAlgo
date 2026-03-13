"""
RoboAlgo — Price Levels API Router

GET /api/price-levels/{symbol}
  Returns multi-source clustered support/resistance zones for any symbol.
  Sources: MAs, Fibonacci, Pivot Points, Gann, BB, ATR, Round Numbers.
"""

from fastapi import APIRouter, HTTPException

from signal_engine.price_levels import PriceLevelEngine

router = APIRouter()
_engine = PriceLevelEngine()


@router.get("/{symbol}")
def get_price_levels(symbol: str):
    """
    Clustered support/resistance zones for a symbol.

    Returns:
      symbol, current_price, atr, atr_pct,
      levels  – up to 24 clusters sorted by strength,
      zones   – named zones: buy_zone, accumulate, stop,
                             scale_in, target, distribution

    Returns empty levels (200) for symbols not in the database — the frontend
    fetches via yfinance for all other data so we simply skip price-level zones
    rather than crashing with a 404.
    """
    result = _engine.compute_levels(symbol.upper())
    if "error" in result:
        # Return empty result so non-DB symbols don't crash the frontend
        return {
            "symbol":        symbol.upper(),
            "current_price": None,
            "atr":           None,
            "atr_pct":       None,
            "levels":        [],
            "zones":         {},
        }
    return result
