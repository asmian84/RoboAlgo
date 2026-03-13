"""Stage 10 — Signal Confidence.

Computes a composite confidence score for every instrument's latest
signal, combining historical reliability, model agreement, regime
match, feature stability, and confluence density.

Pipeline position:
  ... -> Stage 9 Strategy Evolution -> Stage 10 Signal Confidence
"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, SignalConfidence
from signal_confidence_engine.confidence_score import compute_signal_confidence

logger = logging.getLogger("pipeline.stage10_signal_confidence")

UPSERT_COLS = [
    "confidence_score", "confidence_tier",
    "historical_reliability", "model_agreement",
    "regime_match", "feature_stability", "confluence_density",
]


def run(symbol: str | None = None) -> int:
    """Run Stage 10 for one or all instruments."""
    with get_session() as session:
        SignalConfidence.__table__.create(bind=session.bind, checkfirst=True)

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
                logger.warning("Stage 10 failed for %s: %s", inst.symbol, exc)

        logger.info("Stage 10 Signal Confidence: processed %d instruments", count)
        return count


def _process(session, inst: Instrument) -> None:
    """Process a single instrument."""
    result = compute_signal_confidence(inst.symbol)

    components = result.get("components", {})

    record = {
        "instrument_id":        inst.id,
        "date":                 date.today(),
        "confidence_score":     result.get("confidence_score"),
        "confidence_tier":      result.get("confidence_tier"),
        "historical_reliability": components.get("historical_reliability"),
        "model_agreement":      components.get("model_agreement"),
        "regime_match":         components.get("regime_match"),
        "feature_stability":    components.get("feature_stability"),
        "confluence_density":   components.get("confluence_density"),
    }

    stmt = pg_insert(SignalConfidence).values([record])
    stmt = stmt.on_conflict_do_update(
        constraint="uq_sigconf_inst_date",
        set_={col: stmt.excluded[col] for col in UPSERT_COLS},
    )
    session.execute(stmt)
    session.commit()
