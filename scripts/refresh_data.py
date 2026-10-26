"""
Refresh the whole signal chain, in dependency order.

Why this exists: the chain has a strict order, and running one link alone does
nothing visible. Each engine left-joins the one above it and skips bars whose
upstream columns are NaN, so a missing link silently truncates everything below
it at the date it stopped:

    prices -> indicators -> volatility -> compression -> breakout

That is not theoretical. This repo shipped for months with stages 2 and 3
unimplemented, and the result was compression and breakout frozen at an old
date while prices looked current — no error anywhere, just quietly stale
signals. Re-running stage 1 alone never fixed it.

Run it directly:

    python scripts/refresh_data.py            # incremental, the daily case
    python scripts/refresh_data.py --full     # ignore stored state, re-pull

Or on a schedule. Two options, in order of preference:

  1. Celery beat — infrastructure/scheduler.py already schedules the pipeline
     stages, and a worker runs them off the API process where a multi-minute
     job blocks nothing:
         celery -A infrastructure.celery_app worker --loglevel=info
         celery -A infrastructure.scheduler beat --loglevel=info

  2. cron, if you do not want Redis and a worker running. Weekdays after the
     US close (21:30 UTC / 17:30 ET):
         30 21 * * 1-5 cd /path/to/repo && venv/bin/python scripts/refresh_data.py

Do NOT put this back in the API process on a timer. It was there once: a thread
POSTing to /api/pipeline/run every five minutes, regenerating features for ~239
instruments in-process. Uvicorn runs one worker, so the API answered nothing for
roughly two minutes out of every five and every page failed intermittently with
what looked like a frontend bug.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Running this as `python scripts/refresh_data.py` puts scripts/ on sys.path
# rather than the repo root, so the project packages would not import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("refresh_data")


def _step(label: str, fn) -> tuple[str, str, float]:
    """Run one link. Never raises — a later link failing must not hide that the
    earlier ones succeeded, and a partial refresh is still worth keeping."""
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        logger.info("%-22s ok    %s  (%.0fs)", label, result, elapsed)
        return label, "ok", elapsed
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("%-22s FAILED %s: %s  (%.0fs)", label, type(exc).__name__, exc, elapsed)
        return label, f"FAILED {type(exc).__name__}", elapsed


def run(full: bool = False) -> int:
    """Refresh every link in order. Returns the count of failed steps."""
    from pipeline.stage1_prices import run as stage1, freshness
    from pipeline.stage2_indicators import run as stage2
    from pipeline.stage3_volatility import run as stage3
    from range_engine.breakout import BreakoutEngine
    from range_engine.compression import RangeCompressionEngine

    before = freshness(0)
    logger.info("price_data before: %s (%s days old)", before["newest"], before["age_days"])

    outcomes = [
        _step("stage1 prices", lambda: stage1(full=full)),
        _step("stage2 indicators", stage2),
        _step("stage3 volatility", stage3),
        _step("compression", lambda: RangeCompressionEngine().compute_and_store()),
        _step("breakout", lambda: BreakoutEngine().compute_and_store()),
    ]

    after = freshness(0)
    logger.info("price_data after:  %s (%s days old)", after["newest"], after["age_days"])

    failures = [label for label, status, _ in outcomes if status != "ok"]
    if failures:
        logger.error("%d step(s) failed: %s", len(failures), ", ".join(failures))
        logger.error(
            "Anything downstream of the first failure is now stale even if it "
            "reports no error — the chain only advances in order."
        )
    return len(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore stored state and re-pull price history from DATA_START_DATE.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    # These engines are chatty at INFO and drown the step summary.
    for noisy in ("data_engine.downloader", "range_engine", "indicator_engine",
                  "volatility_engine", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return run(full=args.full)


if __name__ == "__main__":
    raise SystemExit(main())
