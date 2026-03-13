"""
RoboAlgo — Celery Worker Configuration

Provides helper functions for starting and configuring workers.
Supports multiple queue configurations for different workload types.

Queues:
    pipeline  — sequential pipeline stages (single worker)
    symbols   — per-symbol parallel processing (multiple workers)
    default   — general tasks

Usage:
    # Start pipeline worker (single process, sequential):
    celery -A infrastructure.celery_app worker -Q pipeline -c 1 --loglevel=info

    # Start symbol workers (parallel, 4 processes):
    celery -A infrastructure.celery_app worker -Q symbols -c 4 --loglevel=info

    # Start all-in-one worker:
    celery -A infrastructure.celery_app worker -Q pipeline,symbols,default -c 2 --loglevel=info

    # Start beat scheduler:
    celery -A infrastructure.celery_app beat -S celery.beat:PersistentScheduler --loglevel=info
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("infrastructure.worker_config")

# Worker configuration profiles
WORKER_PROFILES = {
    "pipeline": {
        "queues": ["pipeline"],
        "concurrency": 1,
        "description": "Sequential pipeline execution (one stage at a time)",
    },
    "symbols": {
        "queues": ["symbols"],
        "concurrency": int(os.getenv("ROBOALGO_SYMBOL_WORKERS", "4")),
        "description": "Parallel per-symbol processing",
    },
    "all": {
        "queues": ["pipeline", "symbols", "default"],
        "concurrency": int(os.getenv("ROBOALGO_WORKERS", "2")),
        "description": "All-in-one worker for development",
    },
}


def get_worker_command(profile: str = "all") -> list[str]:
    """Get the Celery worker command for a given profile.

    Args:
        profile: One of 'pipeline', 'symbols', or 'all'.

    Returns:
        Command as list of strings.
    """
    config = WORKER_PROFILES.get(profile, WORKER_PROFILES["all"])
    queues = ",".join(config["queues"])
    return [
        "celery",
        "-A", "infrastructure.celery_app",
        "worker",
        "-Q", queues,
        "-c", str(config["concurrency"]),
        "--loglevel=info",
    ]


def get_beat_command() -> list[str]:
    """Get the Celery beat scheduler command."""
    return [
        "celery",
        "-A", "infrastructure.celery_app",
        "beat",
        "--loglevel=info",
    ]


def dispatch_parallel_pipeline(
    stage_numbers: list[int],
    symbols: list[str] | None = None,
) -> dict:
    """Dispatch pipeline stages for parallel symbol processing.

    Each stage runs in parallel across all symbols, but stages
    themselves are sequential (stage N must finish before stage N+1).

    Args:
        stage_numbers: List of stage numbers to run.
        symbols: Optional symbol list. None = all instruments.

    Returns:
        Summary of dispatched tasks.
    """
    from celery import chain
    from infrastructure.celery_app import run_parallel_symbols

    tasks = chain(
        run_parallel_symbols.s(stage, symbols)
        for stage in sorted(stage_numbers)
    )
    result = tasks.apply_async()
    return {
        "chain_id": str(result.id),
        "stages": sorted(stage_numbers),
        "status": "dispatched",
    }


def check_redis_connection() -> bool:
    """Check if Redis is reachable."""
    try:
        import redis
        r = redis.from_url(os.getenv("ROBOALGO_REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        return True
    except Exception:
        return False


def get_worker_status() -> dict:
    """Get status of running Celery workers."""
    try:
        from infrastructure.celery_app import app
        inspector = app.control.inspect()
        active = inspector.active() or {}
        stats = inspector.stats() or {}
        return {
            "redis_connected": check_redis_connection(),
            "workers": list(active.keys()),
            "worker_count": len(active),
            "active_tasks": sum(len(tasks) for tasks in active.values()),
            "status": "ok" if active else "no_workers",
        }
    except Exception as exc:
        return {
            "redis_connected": check_redis_connection(),
            "workers": [],
            "worker_count": 0,
            "active_tasks": 0,
            "status": f"error: {exc}",
        }
