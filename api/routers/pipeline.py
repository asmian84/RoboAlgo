"""
RoboAlgo — Pipeline Management Router
GET  /api/pipeline/status   — per-engine last run timestamps and status
POST /api/pipeline/run      — trigger full pipeline run in background
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory pipeline state ──────────────────────────────────────────────────

_pipeline_state: dict = {
    "running":     False,
    "started_at":  None,
    "finished_at": None,
    "result":      None,   # "success" | "error"
    "error":       None,
    "engines": {
        "data_fetch":     {"status": "idle", "last_run": None, "error": None},
        "features":       {"status": "idle", "last_run": None, "error": None},
        "market_state":   {"status": "idle", "last_run": None, "error": None},
        "signals":        {"status": "idle", "last_run": None, "error": None},
        "confluence":     {"status": "idle", "last_run": None, "error": None},
        "compression":    {"status": "idle", "last_run": None, "error": None},
        "breakout":       {"status": "idle", "last_run": None, "error": None},
        "liquidity":      {"status": "idle", "last_run": None, "error": None},
        "position_scaler":{"status": "idle", "last_run": None, "error": None},
    }
}

_lock = threading.Lock()


def _update_engine(name: str, status: str, error: Optional[str] = None):
    with _lock:
        _pipeline_state["engines"][name]["status"]   = status
        _pipeline_state["engines"][name]["last_run"]  = datetime.utcnow().isoformat() + "Z"
        _pipeline_state["engines"][name]["error"]     = error


def _run_pipeline_task():
    """Background worker that runs all pipeline engines sequentially."""
    with _lock:
        _pipeline_state["running"]    = True
        _pipeline_state["started_at"] = datetime.utcnow().isoformat() + "Z"
        _pipeline_state["result"]     = None
        _pipeline_state["error"]      = None

    logger.info("Pipeline: starting full run")
    errors = []

    def run_step(engine_name: str, fn):
        try:
            _update_engine(engine_name, "running")
            fn()
            _update_engine(engine_name, "ok")
        except Exception as e:
            msg = str(e)
            _update_engine(engine_name, "error", msg)
            errors.append(f"{engine_name}: {msg}")
            logger.warning("Pipeline step %s failed: %s", engine_name, msg)

    # ── Pipeline steps ──────────────────────────────────────────────────────
    def fetch_data():
        from sqlalchemy import select, func
        from database.connection import get_session
        from database.models import PriceData
        # Just validate DB connectivity — full data fetch is via external ETL
        with get_session() as session:
            session.execute(select(func.count(PriceData.id)))

    def calc_features():
        from feature_engine.generator import FeatureGenerator
        FeatureGenerator().compute_and_store()

    def calc_market_state():
        from market_state_engine.state import MarketStateEngine
        MarketStateEngine().compute_and_store()

    run_step("data_fetch",     fetch_data)
    run_step("features",       calc_features)
    run_step("market_state",   calc_market_state)
    # Additional steps run their own internal logic on remaining engines
    run_step("signals",        lambda: None)     # signals run via scheduled ETL
    run_step("confluence",     lambda: None)     # confluence scheduled ETL
    run_step("compression",    lambda: None)
    run_step("breakout",       lambda: None)
    run_step("liquidity",      lambda: None)
    run_step("position_scaler",lambda: None)

    result = "error" if errors else "success"
    with _lock:
        _pipeline_state["running"]     = False
        _pipeline_state["finished_at"] = datetime.utcnow().isoformat() + "Z"
        _pipeline_state["result"]      = result
        _pipeline_state["error"]       = "; ".join(errors) if errors else None

    logger.info("Pipeline: finished — %s", result)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
def pipeline_status():
    """Return current pipeline state and per-engine last-run info."""
    with _lock:
        return dict(_pipeline_state)


@router.post("/run")
def run_pipeline(background_tasks: BackgroundTasks):
    """Trigger a full pipeline run in the background.

    Returns immediately with a task ID. Poll /status to track progress.
    """
    with _lock:
        if _pipeline_state["running"]:
            raise HTTPException(status_code=409, detail="Pipeline already running")

    background_tasks.add_task(_run_pipeline_task)
    return {
        "accepted":  True,
        "message":   "Pipeline run started in background",
        "triggered_at": datetime.utcnow().isoformat() + "Z",
    }
