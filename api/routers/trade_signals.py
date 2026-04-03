# api/routers/trade_signals.py
"""
Trade Signal Engine API — scan, lookup, universe endpoints.

Endpoints:
  GET  /scan?profile=SWING&account=10000    — scan top candidates
  GET  /lookup/{symbol}?profile=SWING       — deep single-symbol analysis
  GET  /universe                            — list current universe candidates
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/scan")
def trade_scan(
    profile: str = Query("SWING", description="SCALP | SWING | POSITION"),
    account: float = Query(10000, description="Account size in dollars"),
    max_symbols: int = Query(50, description="Max symbols to deep-scan"),
):
    """Scan top universe candidates and return ranked trade recommendations."""
    try:
        from trade_signal_engine.universe import get_full_universe, get_options_universe
        from trade_signal_engine.scanner import scan_universe

        # Get universe candidates
        candidates = get_full_universe(max_candidates=max_symbols)
        symbols = [c.symbol for c in candidates]
        options_syms = get_options_universe()

        if not symbols:
            return {"error": "No symbols in universe", "recommendations": []}

        result = scan_universe(
            symbols=symbols,
            profile=profile.upper(),
            account_size=account,
            options_symbols=options_syms,
        )
        return result

    except Exception as e:
        logger.error("Trade scan failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lookup/{symbol}")
def trade_lookup(
    symbol: str,
    profile: str = Query("SWING", description="SCALP | SWING | POSITION"),
    account: float = Query(10000, description="Account size in dollars"),
):
    """Deep analysis of a single symbol — returns full trade recommendation."""
    try:
        from trade_signal_engine.scanner import scan_symbol

        result = scan_symbol(
            symbol=symbol.upper(),
            profile=profile.upper(),
            account_size=account,
            fetch_options=True,
        )
        if "error" in result and not result.get("verdict"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Trade lookup failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/universe")
def trade_universe():
    """Return current universe of stock candidates with flags."""
    try:
        from trade_signal_engine.universe import get_full_universe

        candidates = get_full_universe(max_candidates=100)
        return {
            "count": len(candidates),
            "candidates": [
                {
                    "symbol": c.symbol,
                    "source": c.source,
                    "price": c.price,
                    "change_pct": c.change_pct,
                    "volume_ratio": c.volume_ratio,
                    "flags": c.flags,
                    "market_cap": c.market_cap,
                    "short_pct": c.short_pct,
                    "float_shares": c.float_shares,
                }
                for c in candidates
            ],
        }
    except Exception as e:
        logger.error("Universe fetch failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
