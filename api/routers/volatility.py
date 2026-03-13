"""
RoboAlgo — Volatility Regime API
Endpoints for volatility regime data, compression/expansion signals.
"""

from fastapi import APIRouter, Query
from typing import Optional

from volatility_engine.regime import VolatilityRegimeEngine

router = APIRouter()
_engine = VolatilityRegimeEngine()


@router.get("/summary")
def get_regime_summary():
    """
    Latest volatility regime for all instruments grouped by regime.
    Returns {LOW_VOL: [...], NORMAL_VOL: [...], HIGH_VOL: [...]}.
    Useful for dashboard heatmap and instrument filter.
    """
    return _engine.get_regime_summary()


@router.get("/compression")
def get_compressed_instruments():
    """
    All instruments currently in compression (is_compression=True).
    These are the highest-priority setups — compression → expansion = trade signal.
    """
    summary = _engine.get_regime_summary()
    compressed = []
    for regime_list in summary.values():
        for item in regime_list:
            if item.get("is_compression"):
                compressed.append(item)
    # Sort by BB percentile (tightest compression first)
    compressed.sort(key=lambda x: x.get("bb_pct", 1.0))
    return compressed


@router.get("/{symbol}/latest")
def get_latest_regime(symbol: str):
    """
    Latest volatility regime row for a single symbol.
    Returns: regime, is_compression, is_expansion, percentile ranks.
    """
    result = _engine.get_latest_regime(symbol.upper())
    if result is None:
        return {
            "symbol":         symbol.upper(),
            "regime":         "NORMAL_VOL",
            "is_compression": False,
            "is_expansion":   False,
            "message":        "No volatility data — run volatility pipeline first.",
        }
    result["symbol"] = symbol.upper()
    return result


@router.get("/{symbol}")
def get_volatility_history(
    symbol: str,
    limit: int = Query(default=500, ge=1, le=5000),
):
    """
    Full volatility regime history for a symbol.
    Ordered oldest → newest. Use for charting regime overlays.
    """
    return _engine.get_history(symbol.upper(), limit=limit)
