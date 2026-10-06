"""
RoboAlgo — Celery Beat Scheduler Configuration

Defines recurring schedules for pipeline execution.

ALL TIMES ARE AMERICA/NEW_YORK (see infrastructure/celery_app.py, which sets
timezone="America/New_York" and enable_utc=False). They were UTC, which meant
the "after market close" run actually fired at 14:30 ET in summer and 13:30 ET
in winter — before the 16:00 ET close — and stored a partial session as the
day's daily bar.

Schedules:
    - Daily chain (prices -> indicators -> volatility -> range engines):
      weekdays from 16:30 ET, i.e. after the 16:00 ET close
    - Downstream stages (cycles, nodes, physics): weekdays from 16:50 ET
    - Signal confidence (stage 10): weekdays at 17:15 ET
    - Model monitoring (stage 11): daily at 17:30 ET
    - Strategy evolution (stage 9): Sundays at 02:00 ET (weekly)
    - Auto-trader cycle: every 15 minutes during US regular hours, Mon-Fri

A NOTE ON THE 5-MINUTE GAPS. Each entry below assumes the previous one has
finished. That is a guess, not a guarantee — beat fires on the clock and does
not wait. On the current ~234-instrument universe the measured times are
comfortable (prices 38s, indicators 96s, volatility 111s), but the gaps are
not self-correcting: grow the universe and stage N+1 starts reading rows
stage N has not written yet, which does not raise, it just produces less.
If the universe grows materially, chain these with a Celery chord or collapse
them into one task the way run_range_engines already does.
"""

from __future__ import annotations

from celery.schedules import crontab

from infrastructure.celery_app import app


# ── Beat Schedule ─────────────────────────────────────────────────────────────

app.conf.beat_schedule = {
    # ── The chain that feeds the Finder ──────────────────────────────────────
    # prices -> indicators -> volatility -> compression -> breakout.
    # Each link left-joins the one above it and skips bars whose upstream
    # columns are NaN, so a gap anywhere truncates everything below it at the
    # date it stopped, silently. Keep them in this order.
    "daily-prices": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=30, hour=16, day_of_week="1-5"),
        "args": [1],  # Stage 1: price ingestion
        "options": {"queue": "pipeline"},
    },
    "daily-indicators": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=35, hour=16, day_of_week="1-5"),
        "args": [2],  # Stage 2: indicators
        "options": {"queue": "pipeline"},
    },
    "daily-volatility": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=40, hour=16, day_of_week="1-5"),
        "args": [3],  # Stage 3: volatility regimes
        "options": {"queue": "pipeline"},
    },
    # The link that was missing entirely. Compression and breakout are not
    # numbered stages and were in no schedule, so `range_compressions` and
    # `breakout_signals` — the two tables the Finder reads — were never
    # refreshed by anything. Running Celery correctly would not have helped;
    # there was nothing to run. Both engines live in one task so breakout
    # cannot start before compression has written the bars it joins against.
    "daily-range-engines": {
        "task": "run_range_engines",
        "schedule": crontab(minute=45, hour=16, day_of_week="1-5"),
        "options": {"queue": "pipeline"},
    },

    # ── Downstream analytics ─────────────────────────────────────────────────
    # Stages 4 (patterns) and 5 (confluence) are deliberately absent: the
    # modules pipeline/stage4_*.py and pipeline/stage5_*.py do not exist, so
    # scheduling them only produced a "not_implemented" error every weekday.
    # run_pipeline_stage's own error message says to implement the module or
    # remove the entry; this is the removal. Restore the entries here if those
    # modules ever get written — stage_map in celery_app.py still lists them.
    "daily-cycles": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=50, hour=16, day_of_week="1-5"),
        "args": [6],  # Stage 6: cycle projections
        "options": {"queue": "pipeline"},
    },
    "daily-confluence-nodes": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=55, hour=16, day_of_week="1-5"),
        "args": [7],  # Stage 7: confluence nodes
        "options": {"queue": "pipeline"},
    },
    "daily-market-physics": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=0, hour=17, day_of_week="1-5"),
        "args": [8],  # Stage 8: market physics
        "options": {"queue": "pipeline"},
    },

    # Signal confidence — after main pipeline
    "daily-signal-confidence": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=15, hour=17, day_of_week="1-5"),
        "args": [10],  # Stage 10: signal confidence
        "options": {"queue": "pipeline"},
    },

    # Model monitoring — daily
    "daily-model-monitor": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=30, hour=17),
        "args": [11],  # Stage 11: model monitor
        "options": {"queue": "pipeline"},
    },

    # Strategy evolution — weekly on Sunday, well clear of any daily run
    "weekly-strategy-evolution": {
        "task": "run_pipeline_stage",
        "schedule": crontab(minute=0, hour=2, day_of_week=0),
        "args": [9],  # Stage 9: strategy evolution
        "options": {"queue": "pipeline"},
    },

    # Auto-trader cycle — every 15 minutes during US regular hours.
    # 9:30-16:00 ET is the session, but this fires 9:00-16:45 because a
    # crontab cannot express a :30 start and a :00 end in one entry. The
    # cycles outside the session are harmless: auto_trader/loop.py gates on
    # broker.is_market_open() before doing anything and audits the no-op
    # (unless cfg.ignore_market_hours is set). The clock decides, not this
    # schedule. Previously hour="13-20" in UTC — right in summer, an hour
    # early all winter.
    "auto-trader-cycle": {
        "task": "auto_trader_cycle",
        "schedule": crontab(minute="*/15", hour="9-16", day_of_week="1-5"),
        "options": {"queue": "trading"},
    },
}
