"""
RoboAlgo — Celery Beat Scheduler Configuration

Defines recurring schedules for pipeline execution.

Schedules:
    - Daily pipeline (stages 1-8): weekdays at 18:30 UTC (after US market close)
    - Signal confidence (stage 10): weekdays at 19:00 UTC
    - Model monitoring (stage 11): daily at 19:30 UTC
    - Strategy evolution (stage 9): Sundays at 02:00 UTC (weekly)
"""

from __future__ import annotations

from celery.schedules import crontab

from infrastructure.celery_app import app


# ── Beat Schedule ─────────────────────────────────────────────────────────────

app.conf.beat_schedule = {
    # Daily data pipeline — runs Mon-Fri after market close
    "daily-pipeline-prices": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=30, hour=18, day_of_week="1-5"),
        "args": [1],  # Stage 1: price ingestion
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-indicators": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=35, hour=18, day_of_week="1-5"),
        "args": [2],  # Stage 2: indicators
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-volatility": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=40, hour=18, day_of_week="1-5"),
        "args": [3],  # Stage 3: volatility regimes
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-patterns": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=45, hour=18, day_of_week="1-5"),
        "args": [4],  # Stage 4: patterns
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-confluence": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=50, hour=18, day_of_week="1-5"),
        "args": [5],  # Stage 5: confluence scoring
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-cycles": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=55, hour=18, day_of_week="1-5"),
        "args": [6],  # Stage 6: cycle projections
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-nodes": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=0, hour=19, day_of_week="1-5"),
        "args": [7],  # Stage 7: confluence nodes
        "options": {"queue": "pipeline"},
    },
    "daily-pipeline-physics": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=5, hour=19, day_of_week="1-5"),
        "args": [8],  # Stage 8: market physics
        "options": {"queue": "pipeline"},
    },

    # Signal confidence — after main pipeline
    "daily-signal-confidence": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=15, hour=19, day_of_week="1-5"),
        "args": [10],  # Stage 10: signal confidence
        "options": {"queue": "pipeline"},
    },

    # Model monitoring — daily
    "daily-model-monitor": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=30, hour=19),
        "args": [11],  # Stage 11: model monitor
        "options": {"queue": "pipeline"},
    },

    # Strategy evolution — weekly on Sunday
    "weekly-strategy-evolution": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=0, hour=2, day_of_week=0),
        "args": [9],  # Stage 9: strategy evolution
        "options": {"queue": "pipeline"},
    },
}
