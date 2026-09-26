# api/routers/crypto_signals.py
"""
Crypto Trade Signal API — scan, lookup, universe endpoints.

Endpoints:
  GET  /scan?profile=SWING&account=10000    — scan top crypto candidates
  GET  /scan/stream                          — SSE streaming scan with progress
  GET  /universe                             — current top-N crypto universe
  GET  /lookup/{symbol}                      — deep single-symbol crypto analysis
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_symbols(symbols: str, max_symbols: int) -> list[str]:
    from trade_signal_engine.crypto_scanner import get_crypto_universe
    if symbols.strip():
        return [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return get_crypto_universe(limit=max_symbols)


@router.get("/scan")
def crypto_scan(
    profile: str = Query("SWING", description="SCALP | SWING | POSITION"),
    account: float = Query(10000, description="Account size in USD"),
    max_symbols: int = Query(30, description="How many crypto assets to scan"),
    symbols: str = Query("", description="Comma-separated crypto tickers (BTC,ETH,SOL…). Empty = top-N by cap."),
):
    """Scan top-N crypto by market cap and return ranked trade recommendations."""
    try:
        from trade_signal_engine.crypto_scanner import scan_crypto_universe
        sym_list = _resolve_symbols(symbols, max_symbols)
        if not sym_list:
            return {"error": "No crypto symbols resolved", "recommendations": [],
                    "top_trades": [], "watchlist": [], "no_trade": [], "errors": []}
        return scan_crypto_universe(
            symbols=sym_list,
            profile=profile.upper(),
            account_size=account,
        )
    except Exception as e:
        logger.error("Crypto scan failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scan/stream")
def crypto_scan_stream(
    profile: str = Query("SWING", description="SCALP | SWING | POSITION"),
    account: float = Query(10000, description="Account size in USD"),
    max_symbols: int = Query(30, description="How many crypto assets to scan"),
    symbols: str = Query("", description="Comma-separated crypto tickers. Empty = top-N by cap."),
):
    """SSE streaming crypto scan. Same event shapes as /api/trade-signals/scan/stream."""
    from trade_signal_engine.crypto_scanner import stream_scan_crypto
    sym_list = _resolve_symbols(symbols, max_symbols)

    def event_stream():
        for event in stream_scan_crypto(
            sym_list, profile=profile.upper(), account_size=account,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/universe")
def crypto_universe(
    limit: int = Query(30, description="Top-N by market cap"),
):
    """Return the crypto universe (top-N by market cap, CoinGecko with fallback)."""
    from trade_signal_engine.crypto_scanner import get_crypto_universe
    syms = get_crypto_universe(limit=limit)
    return {"count": len(syms), "symbols": syms}


@router.get("/lookup/{symbol}")
def crypto_lookup(
    symbol: str,
    profile: str = Query("SWING"),
    account: float = Query(10000),
):
    """Single-symbol crypto analysis — same scoring as batch scan."""
    try:
        from trade_signal_engine.crypto_scanner import scan_crypto_universe
        result = scan_crypto_universe(
            symbols=[symbol.upper()],
            profile=profile.upper(),
            account_size=account,
        )
        recs = result.get("recommendations", [])
        if not recs:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        return {**recs[0], "regime": result.get("regime", {})}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crypto lookup failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
