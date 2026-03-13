"""
RoboAlgo — Confluence Nodes API Router
Serves price-time confluence decision nodes and heatmap data.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query

router = APIRouter()

_node_cache: dict[str, tuple[list, float]] = {}
_heatmap_cache: dict[str, tuple[dict, float]] = {}
_TTL = 300.0


def _fetch_price_data(symbol: str) -> pd.DataFrame:
    """Delegate to cycles router's data fetcher."""
    from api.routers.cycles import _fetch_price_data
    return _fetch_price_data(symbol)


@router.get("/nodes/{symbol}")
def get_confluence_nodes(symbol: str) -> dict[str, Any]:
    """Get price-time confluence decision nodes for a symbol."""
    sym = symbol.upper()
    now = time.time()

    cached, ts = _node_cache.get(sym, (None, 0.0))
    if cached is not None and (now - ts) < _TTL:
        return {"symbol": sym, "nodes": cached}

    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "nodes": [], "error": "No price data"}

    # Get patterns for confluence
    try:
        from pattern_engine.service import PatternService
        patterns = PatternService().detect_for_symbol(sym, resolution_minutes=0)
    except Exception:
        patterns = []

    from confluence_engine.node_detector import detect_confluence_nodes
    nodes = detect_confluence_nodes(df, symbol=sym, patterns=patterns)

    _node_cache[sym] = (nodes, now)
    return {"symbol": sym, "nodes": nodes}


@router.get("/heatmap/{symbol}")
def get_confluence_heatmap(
    symbol: str,
    bins: int = Query(30, ge=10, le=60, description="Number of price bins"),
) -> dict[str, Any]:
    """Get confluence heatmap data for chart overlay."""
    sym = symbol.upper()
    now = time.time()

    cache_key = f"{sym}_{bins}"
    cached, ts = _heatmap_cache.get(cache_key, (None, 0.0))
    if cached is not None and (now - ts) < _TTL:
        return cached

    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data"}

    from confluence_engine.heatmap import generate_heatmap
    result = generate_heatmap(df, symbol=sym, n_price_bins=bins)

    _heatmap_cache[cache_key] = (result, now)
    return result
