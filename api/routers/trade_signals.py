# api/routers/trade_signals.py
"""
Trade Signal Engine API — scan, lookup, universe endpoints.

Endpoints:
  GET  /scan?profile=SWING&account=10000    — scan top candidates
  GET  /scan/stream?profile=SWING&account=10000 — SSE streaming scan with progress
  GET  /lookup/{symbol}?profile=SWING       — deep single-symbol analysis
  GET  /universe                            — list current universe candidates
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

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


@router.get("/scan/stream")
def trade_scan_stream(
    profile: str = Query("SWING", description="SCALP | SWING | POSITION"),
    account: float = Query(10000, description="Account size in dollars"),
    max_symbols: int = Query(8000, description="Max symbols to deep-scan"),
    symbols: str = Query("", description="Comma-separated symbols (empty = full universe + discovery)"),
    categories: str = Query("", description="Comma-separated categories: bluechip,growth,penny,etfs,etc. (empty = all)"),
    discover: bool = Query(False, description="Pre-screen 6400+ stocks for hidden movers (slower)"),
):
    """
    SSE streaming scan with full-market discovery.

    When discover=True (default):
      1. Detects market regime (bull/bear) from SPY
      2. Pre-screens ALL 6,400+ US stocks for unusual movers (volume spikes, big moves)
      3. Merges discovered movers with curated universe
      4. Auto-adds inverse ETFs / hedges if market is bearish
      5. Deep-scans everything with 18-component scoring, streaming results live
    """
    from trade_signal_engine.universe import OPTIONS_FOCUS, _BROAD_UNIVERSE, get_symbols_for_categories, get_price_range_filter, prefilter_by_price
    from trade_signal_engine.scanner import scan_universe_streaming

    _CAT_NAMES = {
        "bluechip": "Blue Chips",
        "growth":   "Growth",
        "etfs":     "ETFs",
        "penny":    "Penny",
        "under5":   "Under $5",
        "under10":  "$5-$10",
        "under20":  "$10-$20",
        "under50":  "$20-$50",
        "over50":   "Over $50",
    }

    scan_label = ""
    price_range = None

    if symbols.strip():
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        do_discover = False
        scan_label = f"{len(sym_list)} symbol{'s' if len(sym_list) != 1 else ''}"
    elif categories.strip():
        cat_list = [c.strip().lower() for c in categories.split(",") if c.strip()]
        price_range = get_price_range_filter(cat_list)
        do_discover = False

        if price_range:
            # Pre-filter using DB prices (instant, no downloads).
            # Unknown-price symbols pass through to scanner which filters
            # during scoring after its own data collection — single download.
            all_candidates = get_symbols_for_categories(cat_list)
            sym_list = prefilter_by_price(all_candidates, price_range, max_symbols=max_symbols)
        else:
            sym_list = get_symbols_for_categories(cat_list)[:max_symbols]

        cat_label = " + ".join(_CAT_NAMES.get(c, c.title()) for c in cat_list)
        scan_label = f"{cat_label}: {len(sym_list)} symbols"
    else:
        # "All Scan" — fetch full market tickers from exchanges
        from trade_signal_engine.universe import _get_all_exchange_tickers
        all_tickers = _get_all_exchange_tickers()
        # Merge curated universe first (priority), then all exchange tickers
        sym_list = list(dict.fromkeys(OPTIONS_FOCUS + _BROAD_UNIVERSE + all_tickers))[:max_symbols]
        do_discover = False
        scan_label = f"Full market: {len(sym_list)} symbols"

    def event_stream():
        for event in scan_universe_streaming(
            sym_list, profile=profile.upper(), account_size=account,
            discover_movers=do_discover, scan_label=scan_label,
            price_range=price_range,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


@router.get("/multi-lookup/{symbol}")
def trade_multi_lookup(
    symbol: str,
    account: float = Query(10000, description="Account size in dollars"),
):
    """Scan a symbol across all 3 profiles (SCALP/SWING/POSITION) with cross-horizon warnings."""
    try:
        from trade_signal_engine.scanner import scan_symbol_multi

        result = scan_symbol_multi(
            symbol=symbol.upper(),
            account_size=account,
            fetch_options=True,
        )
        return result

    except Exception as e:
        logger.error("Multi-lookup failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/minichart/{symbol}")
def trade_minichart(
    symbol: str,
    bars: int = Query(40, description="Number of bars"),
    tf: str = Query("1d", description="Timeframe: 15m | 1h | 1d"),
):
    """Return recent OHLC bars + Slow Stochastic for mini chart rendering."""
    try:
        import numpy as np

        # Fetch bars — try DB first (fast), fallback to yfinance (no AV rate limits)
        df = None
        sym = symbol.upper()
        need = max(bars + 20, 60)

        def _normalize_yf_columns(frame):
            """Flatten yfinance MultiIndex columns to lowercase strings."""
            if frame is None or frame.empty:
                return frame
            new_cols = []
            for c in frame.columns:
                if isinstance(c, tuple):
                    new_cols.append(str(c[0]).lower() if c[0] else str(c).lower())
                else:
                    new_cols.append(str(c).lower())
            frame.columns = new_cols
            # Drop duplicate columns (yfinance sometimes returns extras)
            frame = frame.loc[:, ~frame.columns.duplicated()]
            return frame

        if tf in ("15m", "1h"):
            # Intraday: yfinance only (DB doesn't store intraday)
            try:
                import yfinance as yf
                period = "5d" if tf == "15m" else "30d"
                df = yf.download(sym, period=period, interval=tf, progress=False, timeout=10)
                df = _normalize_yf_columns(df)
            except Exception:
                pass
        else:
            # Daily: DB first, then yfinance. DataLoader.get_prices()
            # doesn't accept a limit kwarg — we tail to the requested depth
            # after fetching. (Previously this passed limit=need which
            # raised TypeError, silently falling back to a 3-month yfinance
            # window and clipping the chart to ~60 bars.)
            try:
                from data_engine.loader import DataLoader
                loader = DataLoader()
                full_df = loader.get_prices(sym)
                if full_df is not None and not full_df.empty:
                    df = full_df.tail(need)
            except Exception:
                pass
            if df is None or (hasattr(df, 'empty') and df.empty) or len(df) < 20:
                try:
                    import yfinance as yf
                    df = yf.download(sym, period="3mo", interval="1d", progress=False, timeout=10)
                    df = _normalize_yf_columns(df)
                except Exception:
                    pass
        if df is None or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Compute Slow Stochastic (14,3,3)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        n = len(df)

        raw_k = []
        for i in range(n):
            if i < 13:
                raw_k.append(50.0)
                continue
            wh = float(np.max(highs[i-13:i+1]))
            wl = float(np.min(lows[i-13:i+1]))
            denom = wh - wl
            raw_k.append(((closes[i] - wl) / denom * 100) if denom > 0 else 50.0)

        raw_k = np.array(raw_k)
        # Slow %K = SMA(raw_k, 3)
        slow_k = np.convolve(raw_k, np.ones(3)/3, mode='same')
        # %D = SMA(slow_k, 3)
        slow_d = np.convolve(slow_k, np.ones(3)/3, mode='same')

        # Compute MA20 and MA50
        ma20 = np.full(n, np.nan)
        ma50 = np.full(n, np.nan)
        for i in range(19, n):
            ma20[i] = float(np.mean(closes[i-19:i+1]))
        for i in range(49, n):
            ma50[i] = float(np.mean(closes[i-49:i+1]))

        # Take last N bars (cap to available data)
        bars = min(bars, n)
        df_out = df.tail(bars)
        start = n - bars

        candles = []
        for i, (idx, row) in enumerate(df_out.iterrows()):
            gi = start + i
            # Format timestamp for axis labels
            try:
                ts = idx.strftime("%b %d") if tf == "1d" else idx.strftime("%m/%d %H:%M")
            except Exception:
                ts = str(idx)[:10]
            # Unix timestamp in ms for klinecharts
            try:
                ts_ms = int(idx.timestamp() * 1000)
            except Exception:
                ts_ms = 0
            c = {
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
                "v": int(row.get("volume", 0) or 0),
                "k": round(float(slow_k[gi]), 1),
                "d": round(float(slow_d[gi]), 1),
                "t": ts,
                "ts": ts_ms,
            }
            if not np.isnan(ma20[gi]):
                c["m20"] = round(float(ma20[gi]), 2)
            if not np.isnan(ma50[gi]):
                c["m50"] = round(float(ma50[gi]), 2)
            candles.append(c)

        return {
            "symbol": symbol.upper(),
            "candles": candles,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Minichart failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
def cache_stats():
    """Return cache status — age, symbol count, updating flag."""
    try:
        from trade_signal_engine.data_cache import get_cache_stats
        return get_cache_stats()
    except Exception as e:
        logger.error("Cache stats failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/update")
def cache_update(
    max_symbols: int = Query(500, description="Max symbols to update"),
):
    """
    Trigger a background cache update for the full universe.
    Returns immediately — cache updates in background.
    """
    try:
        from trade_signal_engine.data_cache import trigger_background_update, get_cache_stats
        from trade_signal_engine.universe import OPTIONS_FOCUS, _BROAD_UNIVERSE

        sym_list = list(dict.fromkeys(OPTIONS_FOCUS + _BROAD_UNIVERSE))[:max_symbols]
        trigger_background_update(sym_list)

        stats = get_cache_stats()
        return {
            "status": "started",
            "symbols": len(sym_list),
            "current_cache": stats,
        }
    except Exception as e:
        logger.error("Cache update trigger failed: %s", e, exc_info=True)
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
