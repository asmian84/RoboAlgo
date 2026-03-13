"""Master signal confidence orchestrator.

Combines five independent confidence components into a single
composite score with a human-readable tier label.

Weights
-------
  30%  historical_reliability
  25%  model_agreement
  20%  regime_match
  15%  feature_stability
  10%  confluence_density
"""

from __future__ import annotations

import logging

from signal_confidence_engine.historical_reliability import compute_historical_reliability
from signal_confidence_engine.model_agreement import compute_model_agreement
from signal_confidence_engine.regime_match import compute_regime_match
from signal_confidence_engine.feature_stability import compute_feature_stability

logger = logging.getLogger("signal_confidence_engine.confidence_score")

# Component weights
WEIGHTS = {
    "historical_reliability": 0.30,
    "model_agreement":       0.25,
    "regime_match":          0.20,
    "feature_stability":     0.15,
    "confluence_density":    0.10,
}


def _get_confluence_density(symbol: str) -> float:
    """Fetch the latest ConfluenceScore for *symbol* from the DB.

    Returns the ``confluence_score`` field (0-100), or 50.0 as a
    neutral fallback when no data is available.
    """
    try:
        from sqlalchemy import select, desc

        from database.connection import get_session
        from database.models import ConfluenceScore, Instrument

        sym = symbol.upper()
        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(Instrument.symbol == sym)
            ).scalar_one_or_none()
            if inst is None:
                return 50.0

            cs = session.execute(
                select(ConfluenceScore)
                .where(ConfluenceScore.instrument_id == inst.id)
                .order_by(desc(ConfluenceScore.date))
                .limit(1)
            ).scalar_one_or_none()

            if cs is not None and cs.confluence_score is not None:
                return float(cs.confluence_score)
    except Exception:
        pass
    return 50.0


def _tier_label(score: float) -> str:
    """Map a 0-100 confidence score to a human-readable tier."""
    if score >= 80:
        return "Very Reliable"
    if score >= 60:
        return "Reliable"
    if score >= 40:
        return "Moderate"
    return "Weak"


def compute_signal_confidence(symbol: str) -> dict:
    """Compute the composite signal confidence score for *symbol*.

    Returns
    -------
    dict with keys:
        symbol             – uppercased ticker
        confidence_score   – weighted composite 0-100
        confidence_tier    – Very Reliable / Reliable / Moderate / Weak
        components         – dict of individual component scores
    """
    sym = symbol.upper()

    # Compute each component (each returns 0-100)
    hist_rel = compute_historical_reliability(sym)
    model_ag = compute_model_agreement(sym)
    reg_match = compute_regime_match(sym)
    feat_stab = compute_feature_stability(sym)
    conf_dens = _get_confluence_density(sym)

    # Weighted composite
    confidence = (
        WEIGHTS["historical_reliability"] * hist_rel
        + WEIGHTS["model_agreement"] * model_ag
        + WEIGHTS["regime_match"] * reg_match
        + WEIGHTS["feature_stability"] * feat_stab
        + WEIGHTS["confluence_density"] * conf_dens
    )
    confidence = max(0.0, min(100.0, confidence))

    return {
        "symbol": sym,
        "confidence_score": round(confidence, 2),
        "confidence_tier": _tier_label(confidence),
        "components": {
            "historical_reliability": round(hist_rel, 2),
            "model_agreement": round(model_ag, 2),
            "regime_match": round(reg_match, 2),
            "feature_stability": round(feat_stab, 2),
            "confluence_density": round(conf_dens, 2),
        },
    }
