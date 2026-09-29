"""
Stage 3 — volatility regimes.

Referenced by infrastructure/celery_app.py's stage_map but never implemented.
Owns `volatility_regimes`, including atr_percentile and bb_width_percentile.

Those two columns are load-bearing well beyond this table. range_compression
left-joins them and skips any bar where either is NaN, so while this stage was
missing, compression stopped dead at the last date it had data for — and the
breakout engine, which reads compression, stopped with it. Re-running stage 1
alone never fixed that; the chain only advances if stages 2 and 3 run first.

Depends on stage 2 having run: the regime calculation reads indicator rows.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select

from database.connection import get_session
from database.models import Instrument, VolatilityRegime

logger = logging.getLogger(__name__)


def run(symbol: str | None = None, **_kwargs) -> int:
    """Compute and store volatility regimes. Returns instruments processed."""
    from volatility_engine.regime import VolatilityRegimeEngine

    target = symbol.upper() if symbol else None
    logger.info("Stage 3: computing volatility regimes (%s)", target or "all instruments")

    VolatilityRegimeEngine().compute_and_store(target)

    session = get_session()
    try:
        stmt = select(func.count(func.distinct(VolatilityRegime.instrument_id)))
        if target:
            stmt = stmt.join(
                Instrument, Instrument.id == VolatilityRegime.instrument_id
            ).where(Instrument.symbol == target)
        count = session.execute(stmt).scalar() or 0
    finally:
        session.close()

    logger.info("Stage 3: volatility regimes present for %d instruments", count)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"instruments with volatility regimes: {run()}")
