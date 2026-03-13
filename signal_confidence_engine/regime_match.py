"""Regime match scorer for trading signals.

Evaluates whether the current volatility regime is favourable for
the type of signal being generated (e.g. breakout signals score
highest in HIGH_VOL + expansion environments).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("signal_confidence_engine.regime_match")


def compute_regime_match(symbol: str) -> float:
    """Score how well the current regime matches ideal trading conditions.

    Returns a 0-100 score based on the combination of volatility
    regime and structural events (expansion / compression).

    Scoring matrix
    --------------
    HIGH_VOL  + expansion     -> 90  (ideal for breakout trades)
    NORMAL_VOL + compression  -> 80  (building energy)
    HIGH_VOL  + no expansion  -> 60  (volatile but undirected)
    NORMAL_VOL + no compress  -> 55  (neutral)
    LOW_VOL                   -> 40  (poor environment)
    """
    from sqlalchemy import select, desc

    from database.connection import get_session
    from database.models import VolatilityRegime, BreakoutSignal, Instrument

    sym = symbol.upper()

    with get_session() as session:
        # Resolve instrument
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == sym)
        ).scalar_one_or_none()
        if inst is None:
            return 50.0

        # Latest volatility regime
        vol_regime = session.execute(
            select(VolatilityRegime)
            .where(VolatilityRegime.instrument_id == inst.id)
            .order_by(desc(VolatilityRegime.date))
            .limit(1)
        ).scalar_one_or_none()

        if vol_regime is None:
            return 50.0

        regime = (vol_regime.regime or "NORMAL_VOL").upper()
        is_compression = bool(vol_regime.is_compression)
        is_expansion = bool(vol_regime.is_expansion)

        # Also check for a recent breakout signal as an expansion indicator
        if not is_expansion:
            breakout = session.execute(
                select(BreakoutSignal)
                .where(BreakoutSignal.instrument_id == inst.id)
                .order_by(desc(BreakoutSignal.date))
                .limit(1)
            ).scalar_one_or_none()
            if breakout is not None and breakout.triggers_met and breakout.triggers_met >= 2:
                is_expansion = True

        # Scoring matrix
        if regime == "LOW_VOL":
            return 40.0

        if regime == "HIGH_VOL":
            if is_expansion:
                return 90.0
            return 60.0

        # NORMAL_VOL
        if is_compression:
            return 80.0
        return 55.0
