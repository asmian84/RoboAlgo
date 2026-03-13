"""
RoboAlgo — Geometry Engine API Router
Serves Gann angles, fans, Square-of-9/144 levels, and price-time symmetry.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()

_cache: dict[str, tuple[dict, float]] = {}
_TTL = 300.0


@router.get("/{symbol}")
def get_geometry(symbol: str) -> dict[str, Any]:
    """Full geometry analysis: Gann angles, fans, Sq-9, Sq-144, symmetry."""
    sym = symbol.upper()
    now = time.time()

    cached, ts = _cache.get(sym, (None, 0.0))
    if cached is not None and (now - ts) < _TTL:
        return cached

    from api.routers.cycles import _fetch_price_data
    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data available"}

    close = df["close"].astype(float).values
    current_price = float(close[-1])

    # Square-of-9 levels
    try:
        from geometry_engine.square_of_9 import square_of_9_levels
        sq9 = square_of_9_levels(current_price)
    except Exception:
        sq9 = {"support": [], "resistance": []}

    # Square-of-144 levels
    try:
        from geometry_engine.square_of_144 import square_of_144_levels
        sq144 = square_of_144_levels(current_price)
    except Exception:
        sq144 = {"support": [], "resistance": []}

    # Gann fans from significant pivots
    try:
        from geometry_engine.gann_fans import generate_gann_fans
        fans = generate_gann_fans(df, n_fans=2, projection_bars=60)
    except Exception:
        fans = {"fans": [], "overlay_lines": []}

    # Price-time symmetry
    try:
        from geometry_engine.price_time_symmetry import compute_price_time_symmetry
        symmetry = compute_price_time_symmetry(df)
    except Exception:
        symmetry = {"symmetry_score": 0, "symmetry_ratio": 0, "symmetric_swings": []}

    result = {
        "symbol": sym,
        "current_price": current_price,
        "square_of_9": sq9,
        "square_of_144": sq144,
        "gann_fans": fans,
        "price_time_symmetry": symmetry,
    }

    _cache[sym] = (result, now)
    return result


@router.get("/{symbol}/fans")
def get_gann_fans(
    symbol: str,
    n_fans: int = Query(2, ge=1, le=5),
    projection_bars: int = Query(60, ge=10, le=200),
) -> dict[str, Any]:
    """Gann fan projections from significant pivot points."""
    sym = symbol.upper()

    from api.routers.cycles import _fetch_price_data
    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data"}

    from geometry_engine.gann_fans import generate_gann_fans
    result = generate_gann_fans(df, n_fans=n_fans, projection_bars=projection_bars)
    result["symbol"] = sym
    return result


@router.get("/{symbol}/levels")
def get_geometry_levels(symbol: str) -> dict[str, Any]:
    """Square-of-9 and Square-of-144 price levels near current price."""
    sym = symbol.upper()

    from api.routers.cycles import _fetch_price_data
    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data"}

    current_price = float(df["close"].astype(float).values[-1])

    from geometry_engine.square_of_9 import square_of_9_levels
    from geometry_engine.square_of_144 import square_of_144_levels

    return {
        "symbol": sym,
        "current_price": current_price,
        "square_of_9": square_of_9_levels(current_price),
        "square_of_144": square_of_144_levels(current_price),
    }
