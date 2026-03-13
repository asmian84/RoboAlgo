"""Stage 6 — Advanced Cycle Analysis.

Runs FFT, wavelet, and Hilbert phase analysis on each instrument,
stores CycleProjection records with projected peak/trough dates.

Pipeline position:
  Stage 1 Regime → Stage 2 Signals → Stage 3 AI → Stage 4 Patterns
  → Stage 5 Price Levels → ▶ Stage 6 Cycles → Stage 7 Confluence Nodes
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PriceData, CycleProjection
from cycle_engine.cycle_projection import project_cycle

logger = logging.getLogger("pipeline.stage6_cycles")

UPSERT_COLS = [
    "dominant_cycle_length", "cycle_strength", "cycle_phase",
    "fft_cycle_length", "fft_strength",
    "wavelet_cycle_length", "wavelet_strength",
    "hilbert_phase", "hilbert_amplitude",
    "next_peak_date", "next_trough_date",
    "next_peak_price", "next_trough_price",
    "cycle_alignment_score",
]


def run(symbol: str | None = None) -> int:
    """Run Stage 6 for one or all instruments.

    Returns number of instruments processed.
    """
    with get_session() as session:
        CycleProjection.__table__.create(bind=session.bind, checkfirst=True)

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
                logger.warning("Stage 6 failed for %s: %s", inst.symbol, exc)

        logger.info("Stage 6 Cycles: processed %d instruments", count)
        return count


def _process(session, inst: Instrument) -> None:
    """Process a single instrument."""
    rows = session.execute(
        select(PriceData.date, PriceData.open, PriceData.high,
               PriceData.low, PriceData.close, PriceData.volume)
        .where(PriceData.instrument_id == inst.id)
        .order_by(PriceData.date.desc())
        .limit(500)
    ).all()

    if len(rows) < 120:
        return

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    result = project_cycle(df)

    record = {
        "instrument_id": inst.id,
        "date": date.today(),
        **{k: result.get(k) for k in UPSERT_COLS},
    }

    stmt = pg_insert(CycleProjection).values([record])
    stmt = stmt.on_conflict_do_update(
        constraint="uq_cycle_proj_inst_date",
        set_={col: stmt.excluded[col] for col in UPSERT_COLS},
    )
    session.execute(stmt)
    session.commit()
