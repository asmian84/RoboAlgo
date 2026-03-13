"""
RoboAlgo — Confluence API Router
Serves confluence scores, decision traces, and ranked signal lists.
"""

from fastapi import APIRouter, HTTPException, Query
from confluence_engine.score import ConfluenceEngine

router = APIRouter()
_engine = ConfluenceEngine()


@router.get("/score/{symbol}")
def get_confluence_score(symbol: str):
    """Full confluence analysis for a single symbol.

    Returns a gated=True response (200) for symbols not in the database
    so the frontend can gracefully show "no data" without a 404 error.
    """
    result = _engine.score_symbol(symbol)
    if "error" in result:
        return {
            "symbol":           symbol.upper(),
            "confluence_score": 0,
            "signal_tier":      "NONE",
            "gated":            True,
            "error":            result["error"],
        }
    return result


@router.get("/top")
def get_top_signals(
    tier: str = Query("MEDIUM", description="Minimum tier: HIGH, MEDIUM, WATCH"),
    limit: int = Query(20, ge=1, le=100),
):
    """Return top-ranked confluence signals."""
    return {"signals": _engine.get_top_signals(min_tier=tier, limit=limit)}


@router.post("/compute/{symbol}")
def compute_confluence(symbol: str):
    """Recompute and store confluence score for a symbol."""
    count = _engine.compute_and_store(symbol=symbol)
    return {"symbol": symbol, "computed": count}


@router.post("/compute")
def compute_all_confluence():
    """Batch recompute confluence for all instruments."""
    count = _engine.compute_and_store()
    return {"computed": count}
