"""
RoboAlgo — Rocket Scanner API Router
Endpoints:
  GET  /api/rocket/scan          — run full scan, return top N candidates
  GET  /api/rocket/scan/quick    — scan against watchlist only (fast)
  GET  /api/rocket/gex/{symbol}  — GEX profile for a single symbol
  GET  /api/rocket/squeeze/{symbol} — volatility squeeze for a single symbol
  GET  /api/rocket/status        — last scan metadata (time, count, stage pass rates)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-process scan cache ──────────────────────────────────────────────────
_last_scan:       list[dict] = []
_last_scan_time:  float      = 0.0
_scan_in_progress: bool      = False
_SCAN_TTL         = 300      # 5 min before allowing a fresh scan


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run scan in a thread (scanner is CPU-bound, must not block event loop)
# ─────────────────────────────────────────────────────────────────────────────

def _run_scan(
    symbols:      list[str] | None,
    top_n:        int,
    skip_options: bool,
    min_price:    float,
    min_volume:   float,
) -> list[dict]:
    from signal_engine.rocket_scanner import RocketScanner
    from ranking_engine.opportunity_ranker import OpportunityRanker
    from data_engine.alphavantage_fetcher import load_ohlcv_from_db
    from config.settings import get_all_instruments, PRIMARY_WATCHLIST

    syms = symbols or get_all_instruments()

    logger.info("Rocket scan: loading price data for %d symbols …", len(syms))
    price_data = load_ohlcv_from_db(syms, lookback_days=400)

    scanner = RocketScanner(
        min_price        = min_price,
        min_volume       = min_volume,
        top_n            = top_n * 3,     # over-fetch for ranker diversity
        skip_options     = skip_options,
        pattern_workers  = 4,
        options_workers  = 4,
    )
    candidates = scanner.scan(price_data)

    ranker  = OpportunityRanker(top_n=top_n)
    ranked  = ranker.rank(candidates)

    return [OpportunityRanker.to_dict(r) for r in ranked]


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/scan")
async def run_scan(
    background_tasks: BackgroundTasks,
    top_n:        int   = Query(default=20, ge=1, le=50),
    skip_options: bool  = Query(default=False),
    min_price:    float = Query(default=3.0,     ge=0),
    min_volume:   float = Query(default=500_000, ge=0),
    force:        bool  = Query(default=False),
):
    """
    Run the full 5-stage rocket scanner.

    Returns top-N candidates ranked by rocket score.
    Results are cached for 5 minutes; use force=true to bypass cache.
    """
    global _last_scan, _last_scan_time, _scan_in_progress

    now = time.time()
    if not force and _last_scan and (now - _last_scan_time) < _SCAN_TTL:
        return {
            "source":    "cache",
            "scan_age_s": round(now - _last_scan_time, 0),
            "count":     len(_last_scan),
            "results":   _last_scan[:top_n],
        }

    if _scan_in_progress:
        return {
            "source":  "scan_in_progress",
            "message": "Scan is running — try again in 30 s",
            "results": _last_scan[:top_n],
        }

    _scan_in_progress = True
    try:
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _run_scan(None, top_n, skip_options, min_price, min_volume),
        )
        _last_scan      = results
        _last_scan_time = time.time()
        return {
            "source":  "fresh",
            "count":   len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Rocket scan failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _scan_in_progress = False


@router.get("/scan/quick")
async def run_quick_scan(
    top_n:     int  = Query(default=20, ge=1, le=50),
    watchlist: str  = Query(default="primary"),
):
    """
    Quick scan against the configured watchlist only (no DB bulk load).
    Skips options stage for speed.  Completes in ~1-2 s.
    """
    from signal_engine.rocket_scanner import RocketScanner
    from ranking_engine.opportunity_ranker import OpportunityRanker
    from data_engine.alphavantage_fetcher import load_ohlcv_from_db
    from config.settings import PRIMARY_WATCHLIST, UNDERLYING_LEADERS

    syms = PRIMARY_WATCHLIST if watchlist == "primary" else UNDERLYING_LEADERS
    price_data = load_ohlcv_from_db(syms, lookback_days=200)

    scanner  = RocketScanner(top_n=top_n * 2, skip_options=True)
    ranker   = OpportunityRanker(top_n=top_n)

    loop       = asyncio.get_event_loop()
    candidates = await loop.run_in_executor(None, lambda: scanner.scan(price_data))
    ranked     = ranker.rank(candidates)

    return {
        "source":  "quick",
        "count":   len(ranked),
        "results": [OpportunityRanker.to_dict(r) for r in ranked],
    }


@router.get("/gex/{symbol}")
async def get_gex(symbol: str, include_vex: bool = Query(default=True)):
    """
    Return full GEX profile + dealer positioning + optional VEX/Charm for a symbol.
    """
    from data_engine.yahoo_options_fetcher import fetch_options_chain
    from options_engine.gamma_exposure import GammaExposureEngine
    from options_engine.dealer_positioning import DealerPositioningEngine
    from options_engine.vanna_exposure import VannaExposureEngine

    chain = fetch_options_chain(symbol.upper())
    if chain is None:
        raise HTTPException(status_code=404, detail=f"No options data for {symbol}")

    gex_engine = GammaExposureEngine()
    dp_engine  = DealerPositioningEngine()

    profile = gex_engine.compute(chain)
    dp      = dp_engine.compute(profile)

    result = {
        **gex_engine.profile_to_dict(profile),
        "dealer_positioning": DealerPositioningEngine.to_dict(dp),
    }

    if include_vex:
        vex_engine = VannaExposureEngine()
        vex_profile = vex_engine.compute(chain)
        result["vex"] = VannaExposureEngine.to_dict(vex_profile)

    return result


@router.get("/squeeze/{symbol}")
async def get_squeeze(symbol: str, bars: int = Query(default=400, ge=50, le=2000)):
    """
    Return volatility squeeze analysis for a single symbol.
    """
    from data_engine.alphavantage_fetcher import fetch_daily_ohlcv
    from signal_engine.volatility_squeeze import VolatilitySqueezeEngine

    df = fetch_daily_ohlcv(symbol.upper(), bars=bars)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

    engine = VolatilitySqueezeEngine()
    result = engine.compute(
        symbol = symbol.upper(),
        high   = df["high"].to_numpy(dtype=float),
        low    = df["low"].to_numpy(dtype=float),
        close  = df["close"].to_numpy(dtype=float),
    )
    return VolatilitySqueezeEngine.result_to_dict(result)


@router.get("/patterns/{symbol}")
async def get_pivot_patterns(symbol: str, bars: int = Query(default=200, ge=50)):
    """
    Return pivot-based pattern detections for a single symbol.
    """
    from data_engine.alphavantage_fetcher import fetch_daily_ohlcv
    from pattern_engine.pivot_engine import PivotEngine

    df = fetch_daily_ohlcv(symbol.upper(), bars=bars)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

    engine   = PivotEngine()
    patterns = engine.detect(
        high   = df["high"].to_numpy(dtype=float),
        low    = df["low"].to_numpy(dtype=float),
        close  = df["close"].to_numpy(dtype=float),
        volume = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else None,
    )
    return {
        "symbol":   symbol.upper(),
        "count":    len(patterns),
        "patterns": [PivotEngine.result_to_dict(p) for p in patterns],
    }


@router.get("/status")
def get_scan_status():
    """Return metadata about the last completed scan."""
    now = time.time()
    return {
        "last_scan_age_s":  round(now - _last_scan_time, 0) if _last_scan_time else None,
        "cached_count":     len(_last_scan),
        "scan_in_progress": _scan_in_progress,
        "cache_ttl_s":      _SCAN_TTL,
    }
