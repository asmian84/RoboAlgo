"""RoboAlgo — Market State API Router"""

from fastapi import APIRouter, HTTPException
from market_state_engine.state import MarketStateEngine

router = APIRouter()
_engine = MarketStateEngine()


@router.get("/{symbol}")
def get_market_state(symbol: str):
    """
    Get current market state for a symbol.
    Returns: state (COMPRESSION/TREND/EXPANSION/CHAOS), component metrics,
             and size_multiplier for position sizing.
    """
    result = _engine.get_latest(symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"No market state data for {symbol}")
    return {"symbol": symbol.upper(), **result}


@router.get("/")
def get_state_summary():
    """Return count of all instruments by market state."""
    return _engine.get_state_summary()
