"""
RoboAlgo — Signal Confidence API Router
Serves composite signal confidence scores and component breakdowns.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_cache: dict[str, tuple[dict, float]] = {}
_TTL = 300.0


@router.get("/{symbol}")
def get_signal_confidence(symbol: str) -> dict[str, Any]:
    """Compute signal confidence score for a symbol."""
    sym = symbol.upper()
    now = time.time()

    cached, ts = _cache.get(sym, (None, 0.0))
    if cached is not None and (now - ts) < _TTL:
        return cached

    from signal_confidence_engine.confidence_score import compute_signal_confidence

    result = compute_signal_confidence(sym)

    _cache[sym] = (result, now)
    return result
