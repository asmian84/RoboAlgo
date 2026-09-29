"""
api/routers/finder.py
────────────────────────────────────────────────────────────────────────────────
The Finder — confluence screener API. Thin delegation to scanner_engine/finder.py;
no indicator, compression, or breakout logic lives here.

Endpoints:
  GET  /api/finder/presets    — the built-in screening presets (signal / pennyking / maker)
  GET  /api/finder/scan       — FAST path: precomputed DB data over the ~234
                                 tracked instruments (indicators/range_compressions/
                                 breakout_signals). Milliseconds, but that universe
                                 is almost entirely large caps and leveraged ETFs —
                                 the pennyking/maker presets will find little to
                                 nothing here.
  GET  /api/finder/scan/wide  — SLOW path: live, full-market scan (~13,786 US
                                 tickers by default), no DB reads. Minutes, not
                                 milliseconds — this is the path that can actually
                                 find penny/small-cap names. See scanner_engine.
                                 finder.scan_wide for the two-stage design.
  GET  /api/finder/health     — per-table data freshness (stale backfill, made visible)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, replace

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
logger = logging.getLogger(__name__)

PRESET_META = {
    "signal": {
        "label": "Signal",
        "description": "No price filter. Golden cross, consolidation, and clean price action — all defaults on.",
    },
    "pennyking": {
        "label": "PennyKing",
        "description": "Sub-$5 names with real liquidity (500k+ average volume).",
    },
    "maker": {
        "label": "Maker",
        "description": "$5-$20 band, 250k+ average volume.",
    },
}


@router.get("/presets")
def presets():
    from scanner_engine.finder import PRESETS

    return {
        name: {**PRESET_META.get(name, {"label": name, "description": ""}), "criteria": asdict(criteria)}
        for name, criteria in PRESETS.items()
    }


@router.get("/scan")
def scan(
    preset: str = Query("", description="signal | pennyking | maker. Empty = library defaults."),
    limit: int = Query(100, ge=1, le=500),
    min_price: float | None = Query(None),
    max_price: float | None = Query(None),
    min_volume: int | None = Query(None),
    require_golden_cross: bool | None = Query(None),
    require_consolidation: bool | None = Query(None),
    max_gap_pct: float | None = Query(None),
    gap_lookback: int | None = Query(None),
    min_confluence: float | None = Query(None),
    require_breakout_ready: bool | None = Query(None),
    consolidation_lookback: int | None = Query(None),
    min_compression_score: float | None = Query(None),
):
    """
    Run the screener. Pass `preset` for a canned band, or override any
    FinderCriteria field directly (overrides apply on top of the preset when
    both are given, or on top of library defaults when preset is empty).
    """
    from scanner_engine.finder import PRESETS, FinderCriteria
    from scanner_engine.finder import scan as run_scan

    if preset:
        if preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"unknown preset '{preset}'. choices: {sorted(PRESETS)}")
        criteria = PRESETS[preset]
    else:
        criteria = FinderCriteria()

    overrides = {
        k: v
        for k, v in {
            "min_price": min_price,
            "max_price": max_price,
            "min_volume": min_volume,
            "require_golden_cross": require_golden_cross,
            "require_consolidation": require_consolidation,
            "max_gap_pct": max_gap_pct,
            "gap_lookback": gap_lookback,
            "min_confluence": min_confluence,
            "require_breakout_ready": require_breakout_ready,
            "consolidation_lookback": consolidation_lookback,
            "min_compression_score": min_compression_score,
        }.items()
        if v is not None
    }
    if overrides:
        criteria = replace(criteria, **overrides)

    results = run_scan(criteria, limit=limit)
    return {
        "preset": preset or None,
        "criteria": asdict(criteria),
        "count": len(results),
        "results": results,
    }


@router.get("/scan/wide")
def scan_wide(
    preset: str = Query("", description="signal | pennyking | maker. Empty = library defaults."),
    limit: int = Query(100, ge=1, le=500),
    max_deep: int = Query(400, ge=1, le=2000, description="Deep-analyze at most this many stage-1 survivors, most liquid (price*volume) first."),
    min_price: float | None = Query(None),
    max_price: float | None = Query(None),
    min_volume: int | None = Query(None),
    require_golden_cross: bool | None = Query(None),
    require_consolidation: bool | None = Query(None),
    max_gap_pct: float | None = Query(None),
    gap_lookback: int | None = Query(None),
    min_confluence: float | None = Query(None),
    require_breakout_ready: bool | None = Query(None),
    consolidation_lookback: int | None = Query(None),
    min_compression_score: float | None = Query(None),
):
    """
    SLOW path. Live, full-market screen (default: the full ~13,786-symbol US
    ticker universe) — does not depend on the precomputed DB tables `/scan`
    reads, so it is the only path that can find sub-$5 / $5-$20 names.
    Same preset + override semantics as `/scan`; takes minutes, not
    milliseconds. confluence is always skipped/None regardless of
    min_confluence — see scanner_engine.finder.scan_wide.
    """
    from scanner_engine.finder import PRESETS, FinderCriteria
    from scanner_engine.finder import scan_wide as run_scan_wide

    if preset:
        if preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"unknown preset '{preset}'. choices: {sorted(PRESETS)}")
        criteria = PRESETS[preset]
    else:
        criteria = FinderCriteria()

    overrides = {
        k: v
        for k, v in {
            "min_price": min_price,
            "max_price": max_price,
            "min_volume": min_volume,
            "require_golden_cross": require_golden_cross,
            "require_consolidation": require_consolidation,
            "max_gap_pct": max_gap_pct,
            "gap_lookback": gap_lookback,
            "min_confluence": min_confluence,
            "require_breakout_ready": require_breakout_ready,
            "consolidation_lookback": consolidation_lookback,
            "min_compression_score": min_compression_score,
        }.items()
        if v is not None
    }
    if overrides:
        criteria = replace(criteria, **overrides)

    outcome = run_scan_wide(criteria, limit=limit, max_deep=max_deep)
    return {
        "preset": preset or None,
        "criteria": asdict(criteria),
        "count": len(outcome["results"]),
        "results": outcome["results"],
        "stats": outcome["stats"],
    }


@router.get("/health")
def health():
    from scanner_engine.finder import health as check_health

    return check_health()
