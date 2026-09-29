"""
Stage 2 — technical indicators.

Referenced by infrastructure/celery_app.py's stage_map but never implemented,
so every scheduled run raised ImportError. The `indicators` table sat frozen at
2026-03-05 as a result.

That staleness is not contained to this table. Stage 3 (volatility regimes)
left-joins these rows, range compression left-joins stage 3, and the breakout
engine depends on compression — each skipping bars whose upstream columns are
NaN. One missing stage silently truncated the entire signal stack at the date
it stopped.

Computation lives in indicator_engine.IndicatorCalculator; this is the
scheduler-facing wrapper, matching the `run(symbol=None) -> int` contract the
other stages use.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select

from database.connection import get_session
from database.models import Indicator, Instrument

logger = logging.getLogger(__name__)


def run(symbol: str | None = None, **_kwargs) -> int:
    """Compute and store indicators. Returns instruments processed.

    Never raises for a single bad instrument — the calculator logs and
    continues, so one delisted ticker cannot stop the nightly run.
    """
    from indicator_engine.calculator import IndicatorCalculator

    target = symbol.upper() if symbol else None
    logger.info("Stage 2: computing indicators (%s)", target or "all instruments")

    IndicatorCalculator().compute_and_store(target)

    # compute_and_store returns None, so report what actually landed rather
    # than passing a meaningless value up to the Celery task.
    session = get_session()
    try:
        stmt = select(func.count(func.distinct(Indicator.instrument_id)))
        if target:
            stmt = stmt.join(
                Instrument, Instrument.id == Indicator.instrument_id
            ).where(Instrument.symbol == target)
        count = session.execute(stmt).scalar() or 0
    finally:
        session.close()

    logger.info("Stage 2: indicators present for %d instruments", count)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"instruments with indicators: {run()}")
