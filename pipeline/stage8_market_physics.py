"""Stage 8 — Market Physics (Net Force Vectors).

Computes directional force vectors from trend, liquidity, volatility,
cycle, and pattern engines. The net force indicates overall market bias.

Pipeline position:
  … → Stage 7 Confluence Nodes → ▶ Stage 8 Market Physics
  → Stage 9 Strategy Evolution
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PriceData, CycleProjection, MarketForce
from physics_engine.net_force import compute_net_force
from pattern_engine.service import PatternService

logger = logging.getLogger("pipeline.stage8_market_physics")

_pattern_service = PatternService()

UPSERT_COLS = [
    "trend_force", "liquidity_force", "volatility_force",
    "cycle_force", "pattern_force", "net_force", "bias", "force_magnitude",
]


def run(symbol: str | None = None) -> int:
    """Run Stage 8 for one or all instruments."""
    with get_session() as session:
        MarketForce.__table__.create(bind=session.bind, checkfirst=True)

        if symbol:
            instruments = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalars().all()
        else:
            instruments = session.execute(select(Instrument)).scalars().all()

        count = 0
        for inst in instruments:
            try:
                _process(session, inst)
                count += 1
            except Exception as exc:
                logger.warning("Stage 8 failed for %s: %s", inst.symbol, exc)

        logger.info("Stage 8 Market Physics: processed %d instruments", count)
        return count


def _process(session, inst: Instrument) -> None:
    """Process a single instrument."""
    rows = session.execute(
        select(PriceData.date, PriceData.open, PriceData.high,
               PriceData.low, PriceData.close, PriceData.volume)
        .where(PriceData.instrument_id == inst.id)
        .order_by(PriceData.date.desc())
        .limit(320)
    ).all()

    if len(rows) < 50:
        return

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # Get cycle phase from CycleProjection if available
    cycle_phase = 0.0
    cycle_strength = 0.0
    cp = session.execute(
        select(CycleProjection)
        .where(CycleProjection.instrument_id == inst.id)
        .order_by(CycleProjection.date.desc())
        .limit(1)
    ).scalar_one_or_none()
    if cp:
        cycle_phase = float(cp.cycle_phase or 0.0)
        cycle_strength = float(cp.cycle_strength or 0.0)

    # Get patterns
    try:
        patterns = _pattern_service.detect_for_symbol(inst.symbol, resolution_minutes=0)
    except Exception:
        patterns = []

    forces = compute_net_force(
        df,
        cycle_phase=cycle_phase,
        cycle_strength=cycle_strength,
        patterns=patterns,
    )

    record = {
        "instrument_id": inst.id,
        "date": date.today(),
        **{k: forces.get(k) for k in UPSERT_COLS},
    }

    stmt = pg_insert(MarketForce).values([record])
    stmt = stmt.on_conflict_do_update(
        constraint="uq_market_force_inst_date",
        set_={col: stmt.excluded[col] for col in UPSERT_COLS},
    )
    session.execute(stmt)
    session.commit()
