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
    max_symbols: int = Query(30, description="Max symbols to deep-scan"),
    symbols: str = Query("", description="Comma-separated symbols to scan (empty = use defaults)"),
):
    """Scan symbols and return ranked trade recommendations."""
    try:
        from trade_signal_engine.universe import get_options_universe, OPTIONS_FOCUS, _BROAD_UNIVERSE
        from trade_signal_engine.scanner import scan_universe

        # If user provided symbols, use those
        if symbols.strip():
            sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        else:
            # Fast path: use a curated list instead of the slow full-universe download
            # Start with options-focused + popular tickers, capped at max_symbols
            sym_list = list(dict.fromkeys(OPTIONS_FOCUS + _BROAD_UNIVERSE[:80]))[:max_symbols]

        if not sym_list:
            return {"error": "No symbols to scan", "recommendations": [], "top_trades": [], "watchlist": [], "no_trade": [], "errors": []}

        # Don't fetch options in scan mode (too slow) — only in lookup mode
        result = scan_universe(
            symbols=sym_list,
            profile=profile.upper(),
            account_size=account,
            options_symbols=[],
            max_workers=2,
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
def trade_universe(
    full_scan: bool = Query(False, description="If true, run full market scan (slow). Default uses curated list."),
):
    """Return current universe of stock candidates with flags."""
    try:
        if full_scan:
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
        else:
            # Fast path: return curated list without downloading
            from trade_signal_engine.universe import OPTIONS_FOCUS, _BROAD_UNIVERSE
            all_syms = list(dict.fromkeys(OPTIONS_FOCUS + _BROAD_UNIVERSE))
            return {
                "count": len(all_syms),
                "candidates": [
                    {"symbol": s, "source": "curated", "price": 0, "change_pct": 0,
                     "volume_ratio": 0, "flags": [], "market_cap": 0, "short_pct": 0, "float_shares": 0}
                    for s in all_syms
                ],
            }
    except Exception as e:
        logger.error("Universe fetch failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
