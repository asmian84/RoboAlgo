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
    # Timezone
    timezone="UTC",
    enable_utc=True,
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

    try:
        import importlib
        module = importlib.import_module(module_path)
        count = module.run(**kwargs)
        logger.info("Stage %d completed: %d items processed", stage_number, count)
        return {"stage": stage_number, "count": count, "status": "ok"}
    except Exception as exc:
        logger.error("Stage %d failed: %s", stage_number, exc)
        raise self.retry(exc=exc)


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
