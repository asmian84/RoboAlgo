"""
api/routers/lab.py
────────────────────────────────────────────────────────────────────────────────
The Lab — isolated experiment bench. Nothing here feeds the live scanner.

Endpoints:
  GET    /api/lab/personas              — persona roster
  GET    /api/lab/providers             — fundamentals providers + availability
  GET    /api/lab/panel/{symbol}        — run the persona panel vs. the incumbent
  GET    /api/lab/runs                  — recent logged runs
  GET    /api/lab/scorecard             — hit rate per source (persona vs scanner)
  DELETE /api/lab/scorecard             — purge log (all, or ?source=…)
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/personas")
def personas():
    from lab_engine.personas import list_personas
    return {"personas": list_personas()}


@router.get("/providers")
def providers():
    from lab_engine.fundamentals import provider_status, DEFAULT_PROVIDER
    return {"providers": provider_status(), "default": DEFAULT_PROVIDER}


@router.get("/panel/{symbol}")
def panel(
    symbol: str,
    provider: str = Query("yfinance", description="yfinance | financialdatasets"),
    personas: str = Query("", description="Comma-separated persona ids. Empty = all."),
    compare_scanner: bool = Query(True, description="Also run the live scanner for side-by-side"),
    log: bool = Query(True, description="Write this run to the experiment log"),
):
    """
    Run the rule-based persona panel on one symbol.

    Returns each persona's verdict, the panel consensus, and — when
    `compare_scanner` is on — the incumbent scanner's verdict on the same
    symbol so the two can be scored against each other later.
    """
    from lab_engine.fundamentals import get_fundamentals
    from lab_engine.personas import run_panel
    from lab_engine.verdict_log import log_run

    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol required")

    try:
        fundamentals = get_fundamentals(sym, provider=provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    only = [p.strip() for p in personas.split(",") if p.strip()] or None
    result = run_panel(fundamentals, only=only)

    incumbent = _scanner_verdict(sym) if compare_scanner else None

    run_id = uuid.uuid4().hex[:12]
    logged = 0
    if log:
        rows = [
            {
                "source": f"persona.{v['id']}",
                "verdict": v["verdict"],
                "score": v["score"],
                "confidence": v["confidence"],
                "meta": {"style": v["style"]},
            }
            for v in result["verdicts"]
        ]
        rows.append({
            "source": "panel.consensus",
            "verdict": result["consensus"],
            "score": result["avg_score"],
            "confidence": None,
            "meta": result["tally"],
        })
        if incumbent and incumbent.get("verdict"):
            rows.append({
                "source": "scanner.incumbent",
                "verdict": _normalize_verdict(incumbent["verdict"]),
                "score": incumbent.get("score"),
                "confidence": None,
                "meta": {
                    "direction": incumbent.get("direction"),
                    "raw_verdict": incumbent.get("verdict"),
                    "confidence_label": incumbent.get("confidence_label"),
                },
            })
        logged = log_run(run_id, sym, rows, provider, fundamentals.get("price"))

    return {
        "run_id": run_id,
        "symbol": sym,
        "provider": provider,
        "fundamentals": fundamentals,
        "data_quality": {
            "fields_present": fundamentals.get("_fields_present", 0),
            "error": fundamentals.get("_error"),
        },
        **result,
        "incumbent": incumbent,
        "logged_rows": logged,
    }


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _label(v) -> str | None:
    return str(v).upper() if v is not None else None


# The scanner speaks LONG/SHORT/NO_TRADE; personas speak BULLISH/BEARISH/NEUTRAL.
# Normalize so the scorecard marks both on one scale.
_VERDICT_MAP = {
    "LONG": "BULLISH", "BUY": "BULLISH",
    "SHORT": "BEARISH", "SELL": "BEARISH",
    "HOLD": "NEUTRAL", "NO_TRADE": "NEUTRAL", "NEUTRAL": "NEUTRAL",
}


def _normalize_verdict(v: str | None) -> str | None:
    if not v:
        return None
    return _VERDICT_MAP.get(v.upper(), v.upper())


def _scanner_verdict(symbol: str) -> dict | None:
    """
    Ask the live scanner what IT thinks about this symbol, for side-by-side.

    Returns None if the scanner can't score it — the panel result is still
    valid on its own, just without a comparison baseline.
    """
    try:
        from trade_signal_engine.collector import collect_batch
        from trade_signal_engine.scorer import UnifiedScorer

        sds = collect_batch([symbol], profile="SWING", fetch_options_for=[], max_workers=1)
        if not sds or sds[0].error:
            return {"verdict": None, "error": sds[0].error if sds else "no data"}

        signal = UnifiedScorer().score(sds[0])
        verdict = getattr(signal, "verdict", None) or getattr(signal, "direction", None)
        return {
            "verdict": str(verdict).upper() if verdict else None,
            "score": _num(getattr(signal, "score", None)),
            # The scanner reports confidence as a band ("LOW"/"MED"/"HIGH"),
            # not a number — keep it as a label rather than forcing a float.
            "confidence_label": _label(getattr(signal, "confidence", None)),
            "direction": getattr(signal, "direction", None),
        }
    except Exception as exc:
        logger.warning("incumbent scanner comparison failed for %s: %s", symbol, exc)
        return {"verdict": None, "error": str(exc)}


@router.get("/runs")
def runs(limit: int = Query(50, ge=1, le=500)):
    from lab_engine.verdict_log import recent_runs, db_path
    return {"runs": recent_runs(limit=limit), "db_path": db_path()}


@router.get("/scorecard")
def scorecard(min_age_hours: float = Query(0.0, ge=0.0, description="Ignore verdicts younger than this")):
    from lab_engine.verdict_log import scorecard as build
    return build(min_age_hours=min_age_hours)


@router.delete("/scorecard")
def purge_scorecard(source: str = Query("", description="Only this source. Empty = purge all.")):
    from lab_engine.verdict_log import purge
    removed = purge(source or None)
    return {"removed": removed, "source": source or "ALL"}
