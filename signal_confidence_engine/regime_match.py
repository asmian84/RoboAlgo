"""Regime match scorer for trading signals.

Evaluates whether the current volatility regime is favourable for
the type of signal being generated (e.g. breakout signals score
highest in HIGH_VOL + expansion environments).

Adjusts the base score by a market-wide *risk-appetite* factor derived
from BTC perpetual-futures funding rate + BTC-vs-SPY correlation break.
Stretched funding is a positioning warning that historically front-runs
equity de-risking — discount long-biased confidence when it fires.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("signal_confidence_engine.regime_match")


def _crypto_risk_adjustment() -> tuple[float, str | None]:
    """
    Return a (multiplier, note) pair to apply to the base regime-match score.

    Multipliers are conservative — the crypto layer nudges confidence, it does
    not override the base scoring matrix. Nothing is applied when the crypto
    layer is unreachable (returns (1.0, None)).

      STRETCHED_LONG   → ×0.85  (positioning stretched — discount confidence)
      STRETCHED_SHORT  → ×1.05  (capitulation — mild boost for setups)
      DIVERGE          → ×0.92  (BTC-SPY correlation broken — regime uncertainty)
      CONFIRM_LONG/SHORT/NEUTRAL → ×1.00 (base score unchanged)
    """
    try:
        from data_engine.crypto_fetcher import compute_crypto_risk_read
        crypto = compute_crypto_risk_read()
    except Exception as exc:
        logger.debug("crypto risk overlay unavailable: %s", exc)
        return 1.0, None

    risk = crypto.get("risk_signal", "NEUTRAL")
    funding_pct = crypto.get("funding_rate_annualized_pct")
    if risk == "STRETCHED":
        state = crypto.get("funding_state", "NEUTRAL")
        if state == "STRETCHED_LONG":
            return 0.85, f"funding stretched long at {funding_pct:.1f}% — discount ×0.85"
        if state == "STRETCHED_SHORT":
            return 1.05, f"funding stretched short at {funding_pct:.1f}% — boost ×1.05"
    if risk == "DIVERGE":
        return 0.92, "BTC-SPY correlation broken — discount ×0.92"
    return 1.0, None


def compute_regime_match(symbol: str) -> float:
    """Score how well the current regime matches ideal trading conditions.

    Returns a 0-100 score based on the combination of volatility
    regime and structural events (expansion / compression), then adjusts
    for market-wide risk appetite (BTC funding + correlation break).

    Scoring matrix (base)
    ---------------------
    HIGH_VOL  + expansion     -> 90  (ideal for breakout trades)
    NORMAL_VOL + compression  -> 80  (building energy)
    HIGH_VOL  + no expansion  -> 60  (volatile but undirected)
    NORMAL_VOL + no compress  -> 55  (neutral)
    LOW_VOL                   -> 40  (poor environment)

    Crypto risk-appetite multiplier — see `_crypto_risk_adjustment()`.
    """
    from sqlalchemy import select, desc

    from database.connection import get_session
    from database.models import VolatilityRegime, BreakoutSignal, Instrument

    sym = symbol.upper()
    base = 50.0

    with get_session() as session:
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == sym)
        ).scalar_one_or_none()
        if inst is None:
            return _apply_crypto_adjustment(base, sym)

        vol_regime = session.execute(
            select(VolatilityRegime)
            .where(VolatilityRegime.instrument_id == inst.id)
            .order_by(desc(VolatilityRegime.date))
            .limit(1)
        ).scalar_one_or_none()

        if vol_regime is None:
            return _apply_crypto_adjustment(base, sym)

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

        if regime == "LOW_VOL":
            base = 40.0
        elif regime == "HIGH_VOL":
            base = 90.0 if is_expansion else 60.0
        else:  # NORMAL_VOL
            base = 80.0 if is_compression else 55.0

    return _apply_crypto_adjustment(base, sym)


def _apply_crypto_adjustment(base: float, symbol: str) -> float:
    """Apply the crypto risk-appetite multiplier and log when it fires."""
    mult, note = _crypto_risk_adjustment()
    if mult == 1.0:
        return round(base, 2)
    adjusted = max(0.0, min(100.0, base * mult))
    if note:
        logger.info(
            "regime_match[%s]: base=%.1f → adjusted=%.1f (%s)",
            symbol, base, adjusted, note,
        )
    return round(adjusted, 2)
