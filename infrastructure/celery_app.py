"""
RoboAlgo — Celery Application Configuration

Provides distributed task execution for pipeline stages.
Each pipeline stage is wrapped as a Celery task for parallel symbol processing.

Usage:
    # Start worker:
    celery -A infrastructure.celery_app worker --loglevel=info --concurrency=4

    # Start scheduler (beat):
    celery -A infrastructure.celery_app beat --loglevel=info
"""

from __future__ import annotations

import logging
import os

from celery import Celery

logger = logging.getLogger("infrastructure.celery")

# Redis connection — default to localhost, configurable via environment
REDIS_URL = os.getenv("ROBOALGO_REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "roboalgo",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone — market local, deliberately not UTC.
    #
    # Every schedule in this system is anchored to the US session (run after
    # the close, trade during regular hours), and the US session moves against
    # UTC twice a year. The beat schedule was written in UTC with a comment
    # claiming it ran "after US market close": 18:30 UTC is 14:30 ET in summer
    # and 13:30 ET in winter, so it fired 90 minutes to 2.5 hours BEFORE the
    # 16:00 ET close and wrote a partial session as that day's daily bar.
    #
    # Expressing schedules in America/New_York makes the crontabs mean what
    # they say and tracks DST on its own. Nothing here uses eta=, so this
    # affects beat scheduling only.
    timezone="America/New_York",
    enable_utc=False,
    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Results
    result_expires=3600,  # 1 hour
    # Retry
    task_default_retry_delay=30,
    task_max_retries=3,
    # Task routing
    task_routes={
        "infrastructure.celery_app.run_pipeline_stage": {"queue": "pipeline"},
        "infrastructure.celery_app.run_symbol_stage": {"queue": "symbols"},
        "auto_trader_cycle": {"queue": "trading"},
    },
)


# ── Pipeline Tasks ────────────────────────────────────────────────────────────


@app.task(bind=True, name="run_pipeline_stage", max_retries=2)
def run_pipeline_stage(self, stage_number: int, **kwargs) -> dict:
    """Run a complete pipeline stage.

    Args:
        stage_number: Pipeline stage (1-11).
        **kwargs: Passed to the stage's run() function.

    Returns:
        dict with stage number, count processed, and status.
    """
    stage_map = {
        1: "pipeline.stage1_prices",
        2: "pipeline.stage2_indicators",
        3: "pipeline.stage3_volatility",
        4: "pipeline.stage4_patterns",
        5: "pipeline.stage5_confluence",
        6: "pipeline.stage6_cycles",
        7: "pipeline.stage7_confluence_nodes",
        8: "pipeline.stage8_market_physics",
        9: "pipeline.stage9_strategy_evolution",
        10: "pipeline.stage10_signal_confidence",
        11: "pipeline.stage11_model_monitor",
    }

    module_path = stage_map.get(stage_number)
    if not module_path:
        return {"stage": stage_number, "error": f"Unknown stage: {stage_number}"}

    import importlib

    # A stage listed in stage_map whose module does not exist is a permanent
    # configuration fault, not a transient one — retrying cannot make the file
    # appear. Stages 2-5 are in this state today, and burning the retry budget
    # on them buried the real message ("this stage was never implemented")
    # under three identical stack traces per scheduled run.
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        logger.error(
            "Stage %d is scheduled but not implemented: %s is missing (%s). "
            "Not retrying — implement the module or remove the stage from "
            "infrastructure/scheduler.py's beat_schedule.",
            stage_number, module_path, exc,
        )
        return {
            "stage": stage_number,
            "status": "not_implemented",
            "module": module_path,
            "error": str(exc),
        }

    try:
        count = module.run(**kwargs)
        logger.info("Stage %d completed: %d items processed", stage_number, count)
        return {"stage": stage_number, "count": count, "status": "ok"}
    except Exception as exc:
        logger.error("Stage %d failed: %s", stage_number, exc)
        raise self.retry(exc=exc)


@app.task(bind=True, name="run_range_engines", max_retries=2)
def run_range_engines(self, symbol: str | None = None) -> dict:
    """Recompute range compression and breakout signals, in that order.

    These two are not in `stage_map` and never were, which meant nothing
    scheduled them: beat ran stages 1-11 and stopped. `range_compressions` and
    `breakout_signals` are exactly the two tables the Finder reads, so the
    screener served frozen results indefinitely while prices and indicators
    updated around it — no error, just a date that never moved.

    They are one task rather than two beat entries because breakout left-joins
    compression and skips bars whose compression columns are NaN. Run breakout
    first, or before compression has finished, and it silently produces nothing
    for the new bars. A single task makes that ordering structural instead of a
    guess about how long the previous entry takes.
    """
    from range_engine.breakout import BreakoutEngine
    from range_engine.compression import RangeCompressionEngine

    try:
        compressions = RangeCompressionEngine().compute_and_store(symbol)
        breakouts = BreakoutEngine().compute_and_store(symbol)
    except Exception as exc:
        logger.error("Range engines failed: %s", exc)
        raise self.retry(exc=exc)

    logger.info(
        "Range engines completed: %d compressions, %d breakout signals",
        compressions, breakouts,
    )
    return {
        "compressions": compressions,
        "breakouts": breakouts,
        "status": "ok",
    }


@app.task(bind=True, name="run_symbol_stage", max_retries=3)
def run_symbol_stage(self, stage_number: int, symbol: str) -> dict:
    """Run a pipeline stage for a single symbol (parallel per-symbol execution).

    Args:
        stage_number: Pipeline stage number.
        symbol: Instrument symbol to process.

    Returns:
        dict with symbol, stage, and result.
    """
    try:
        result = run_pipeline_stage(stage_number, symbol=symbol)
        return {"symbol": symbol, **result}
    except Exception as exc:
        logger.error("Symbol stage %d/%s failed: %s", stage_number, symbol, exc)
        raise self.retry(exc=exc)


@app.task(name="run_full_pipeline")
def run_full_pipeline(stages: list[int] | None = None) -> dict:
    """Run the full pipeline sequentially (stages 1-11).

    Args:
        stages: Optional subset of stages to run. Default = all.

    Returns:
        dict with results per stage.
    """
    all_stages = stages or list(range(1, 12))
    results = {}

    for stage in sorted(all_stages):
        try:
            result = run_pipeline_stage(stage)
            results[stage] = result
        except Exception as exc:
            results[stage] = {"stage": stage, "error": str(exc), "status": "failed"}
            logger.error("Full pipeline failed at stage %d: %s", stage, exc)
            # Continue to next stage — don't halt entire pipeline

    return {"stages": results, "status": "completed"}


@app.task(name="run_parallel_symbols")
def run_parallel_symbols(stage_number: int, symbols: list[str] | None = None) -> dict:
    """Run a stage in parallel across multiple symbols using Celery group.

    Args:
        stage_number: Which pipeline stage.
        symbols: Symbols to process. If None, fetches all from DB.

    Returns:
        dict with task IDs for monitoring.
    """
    from celery import group

    if symbols is None:
        from database.connection import get_session
        from database.models import Instrument
        from sqlalchemy import select

        with get_session() as session:
            instruments = session.execute(select(Instrument.symbol)).scalars().all()
            symbols = list(instruments)

    job = group(
        run_symbol_stage.s(stage_number, sym) for sym in symbols
    )
    result = job.apply_async()
    return {
        "stage": stage_number,
        "symbols_count": len(symbols),
        "group_id": str(result.id),
        "status": "dispatched",
    }


# ── Auto-Trader Task ─────────────────────────────────────────────────────────


@app.task(name="auto_trader_cycle", bind=True, max_retries=0, acks_late=False)
def auto_trader_cycle(self, cycle_id: str | None = None) -> dict:
    """Run a single auto-trader cycle (scan, decide, place orders).

    max_retries=0 and acks_late=False are load-bearing overrides of the
    global task_acks_late=True setting above. If this task were acked late
    (the default here), a worker crash mid-cycle would cause the broker to
    redeliver the task to another worker, which would re-run the trading
    cycle and place a second set of orders on top of whatever the crashed
    worker already submitted. Combined with the Redis cycle lock below and
    the deterministic client_order_id generated inside auto_trader.loop,
    this is one of three independent layers preventing a double-fill.

    Args:
        cycle_id: Optional explicit cycle id. If omitted, a new one is
            generated from the configured auto-trader mode.

    Returns:
        dict with the cycle result, or {"status": "locked"} if another
        worker already holds the cycle lock.
    """
    from auto_trader import audit, state
    from auto_trader.loop import run_cycle

    try:
        from auto_trader import config
        mode = config.load().mode
    except Exception:
        mode = "paper"

    cid = cycle_id or audit.new_cycle_id(mode)

    if not state.acquire_cycle_lock(cid, ttl=900):
        return {"status": "locked"}

    try:
        return run_cycle(cid)
    finally:
        state.release_cycle_lock(cid)
