"""Signal endpoints — confidence-tiered trade plans with market phase."""

from fastapi import APIRouter
from sqlalchemy import select

from database.connection import get_session
from database.models import Instrument, Signal
from signal_engine.generator import SignalGenerator

router = APIRouter()
generator = SignalGenerator()


def _sig_to_dict(symbol: str, s: Signal) -> dict:
    return {
        "symbol":               symbol,
        "date":                 str(s.date),
        "probability":          s.probability,
        "confidence_tier":      s.confidence_tier,
        "market_phase":         s.market_phase,
        "buy_price":            s.buy_price,
        "accumulate_price":     s.accumulate_price,
        "scale_price":          s.scale_price,
        "sell_price":           s.sell_price,
        # Regime-aware v2 fields
        "market_state":         s.market_state,
        "strategy_mode":        s.strategy_mode,
        "setup_quality_score":  s.setup_quality_score,
        "decision_trace":       s.decision_trace,
    }


@router.get("/latest")
def get_latest_signals(
    min_probability: float = 0.0,
    tier: str | None = None,
):
    """
    Latest signal per instrument.
    Optional filters: min_probability (0-1), tier (HIGH/MEDIUM/LOW).
    """
    return generator.get_latest_signals(min_probability=min_probability, tier=tier)


@router.get("/summary")
def get_signal_summary():
    """Count of current signals by tier and market phase."""
    all_sigs = generator.get_latest_signals(min_probability=0.0)
    tiers: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    phases: dict[str, int] = {}
    for s in all_sigs:
        t = s.get("confidence_tier") or ""
        if t in tiers:
            tiers[t] += 1
        p = s.get("market_phase") or "Unknown"
        phases[p] = phases.get(p, 0) + 1
    return {"total": len(all_sigs), "by_tier": tiers, "by_phase": phases}


@router.get("/{symbol}")
def get_signals_for_symbol(symbol: str, limit: int = 100):
    """All signals for a specific instrument, newest first."""
    session = get_session()
    try:
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == symbol.upper())
        ).scalar()
        if not inst:
            return []

        rows = session.execute(
            select(Signal)
            .where(Signal.instrument_id == inst.id)
            .order_by(Signal.date.desc())
            .limit(limit)
        ).scalars().all()

        return [_sig_to_dict(inst.symbol, s) for s in rows]
    finally:
        session.close()
