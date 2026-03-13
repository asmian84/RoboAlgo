"""RoboAlgo — Breakout Signals API Router"""

from fastapi import APIRouter, HTTPException, Query
from range_engine.breakout import BreakoutEngine

router = APIRouter()
_engine = BreakoutEngine()


@router.get("/latest/{symbol}")
def get_latest_breakout(symbol: str):
    result = _engine.get_latest_breakout(symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"No breakout data for {symbol}")
    return result


@router.get("/active")
def get_active_breakouts(min_strength: float = Query(50.0, ge=0, le=100)):
    """Return all instruments with active breakouts above the strength threshold."""
    return {"breakouts": _engine.get_active_breakouts(min_strength)}
