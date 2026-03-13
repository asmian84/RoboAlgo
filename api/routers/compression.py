"""RoboAlgo — Range Compression API Router"""

from fastapi import APIRouter, HTTPException
from range_engine.compression import RangeCompressionEngine

router = APIRouter()
_engine = RangeCompressionEngine()


@router.get("/latest/{symbol}")
def get_compression(symbol: str):
    result = _engine.get_latest(symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"No compression data for {symbol}")
    return result


@router.get("/compressed")
def get_all_compressed():
    """Return all instruments currently in compression."""
    return {"instruments": _engine.get_compressed_instruments()}


@router.get("/mtf/{symbol}")
def get_mtf_compression(symbol: str):
    """Live multi-timeframe compression score (fetches intraday from yfinance)."""
    return _engine.get_mtf_compression(symbol)
