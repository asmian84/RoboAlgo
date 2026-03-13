"""
RoboAlgo — Market Force API Router
Serves market physics net force vectors.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from fastapi import APIRouter

router = APIRouter()

_cache: dict[str, tuple[dict, float]] = {}
_TTL = 300.0


@router.get("/{symbol}")
def get_market_force(symbol: str) -> dict[str, Any]:
    """Compute market force vectors for a symbol."""
    sym = symbol.upper()
    now = time.time()

    cached, ts = _cache.get(sym, (None, 0.0))
    if cached is not None and (now - ts) < _TTL:
        return cached

    # Fetch price data
    from api.routers.cycles import _fetch_price_data
    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data available"}

    # Get cycle phase if available
    cycle_phase = 0.0
    cycle_strength = 0.0
    try:
        from cycle_engine.cycle_projection import project_cycle
        cycle_data = project_cycle(df)
        cycle_phase = cycle_data.get("cycle_phase", 0.0)
        cycle_strength = cycle_data.get("cycle_strength", 0.0)
    except Exception:
        pass

    # Get patterns
    try:
        from pattern_engine.service import PatternService
        patterns = PatternService().detect_for_symbol(sym, resolution_minutes=0)
    except Exception:
        patterns = []

    from physics_engine.net_force import compute_net_force
    forces = compute_net_force(
        df,
        cycle_phase=cycle_phase,
        cycle_strength=cycle_strength,
        patterns=patterns,
    )
    forces["symbol"] = sym

    _cache[sym] = (forces, now)
    return forces
