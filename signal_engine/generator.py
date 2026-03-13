"""
RoboAlgo - Signal Engine
Generates tiered trading signals using probability model + 6-phase market classifier.

v2 additions (regime playbook):
  - Before emitting a signal, the current market regime is retrieved and
    compared against the playbook. Signals incompatible with the active
    strategy are discarded.
  - Every signal now carries: market_state, strategy_mode,
    setup_quality_score, and a human-readable decision_trace.

Confidence Tiers:
  HIGH   probability >= 90%
  MEDIUM probability 70-90%
  LOW    probability 50-70%
  <50%   discarded

Trade Plan (ATR-based):
  buy_price        = current close
  accumulate_price = close - 1 ATR
  scale_price      = close + 2 ATR
  sell_price       = close + 4 ATR
"""

import logging

import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from config.settings import SIGNAL_PARAMS
from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, Feature, Signal
from probability_engine.classifier import ProbabilityClassifier, FEATURE_COLS
from regime_engine.market_phase import (
    classify_phase, get_confidence_tier, signal_qualifies,
)
from strategy_engine.regime_playbook import (
    get_rule, signal_allowed, build_decision_trace, get_active_strategy,
)

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates regime-gated tiered trading signals with full decision traces."""

    def __init__(self):
        self.classifier = ProbabilityClassifier()

    # ── generate_for_symbol ──────────────────────────────────────────────────

    def generate_for_symbol(self, symbol: str) -> list[dict]:
        """
        Generate signals for a single instrument.

        For each qualifying probability date:
          1. Look up current market regime from MarketState table.
          2. Compute SetupQualityScore (live, from scorer).
          3. Gate the signal through the regime playbook.
          4. Attach market_state, strategy_mode, setup_quality_score,
             and a full decision_trace.
        """
        session = get_session()
        try:
            instrument_id = session.execute(
                select(Instrument.id).where(Instrument.symbol == symbol)
            ).scalar()
            if instrument_id is None:
                return []

            features = pd.read_sql(
                select(Feature.date, *[getattr(Feature, c) for c in FEATURE_COLS])
                .where(Feature.instrument_id == instrument_id)
                .order_by(Feature.date),
                session.bind,
            )
            if features.empty:
                return []
            features["date"] = pd.to_datetime(features["date"])
            features = features.set_index("date")

            prices = pd.read_sql(
                select(PriceData.date, PriceData.close)
                .where(PriceData.instrument_id == instrument_id)
                .order_by(PriceData.date),
                session.bind,
            )
            prices["date"] = pd.to_datetime(prices["date"])
            prices = prices.set_index("date")

            indicators = pd.read_sql(
                select(Indicator.date, Indicator.atr)
                .where(Indicator.instrument_id == instrument_id)
                .order_by(Indicator.date),
                session.bind,
            )
            indicators["date"] = pd.to_datetime(indicators["date"])
            indicators = indicators.set_index("date")

            probas = self.classifier.predict(features)
            if probas.empty:
                return []

            # ── Regime (single call per symbol, not per date) ────────────────
            regime      = self._get_market_state(session, instrument_id)
            rule        = get_rule(regime)
            strat       = get_active_strategy(regime)
            quality_score = self._get_setup_quality_score(symbol)

            p = SIGNAL_PARAMS
            signals = []

            for dt in probas.index:
                prob = float(probas.loc[dt])

                # Phase classification (legacy probabilistic tier check)
                feat_row = features.loc[dt].to_dict() if dt in features.index else {}
                phase, _ = classify_phase(feat_row)

                tier = get_confidence_tier(prob)
                if tier is None:
                    continue
                if not signal_qualifies(prob, phase):
                    continue

                # ── Regime playbook gate ──────────────────────────────────────
                volume_ratio = float(feat_row.get("volume_ratio") or 1.0)
                setup_type   = self._infer_setup_type(feat_row, phase)

                # Retrieve latest confluence score for gate check
                confluence = self._get_confluence_score(session, instrument_id, dt)

                allowed, gate_reason = signal_allowed(
                    regime        = regime,
                    setup_type    = setup_type,
                    confluence    = confluence,
                    quality_score = quality_score,
                    volume_ratio  = volume_ratio,
                )

                if not allowed:
                    logger.debug(
                        "%s: signal blocked for %s — %s", symbol, dt.date(), gate_reason
                    )
                    continue

                # ── Prices ───────────────────────────────────────────────────
                close_val = prices.loc[dt, "close"] if dt in prices.index else None
                atr_val   = indicators.loc[dt, "atr"] if dt in indicators.index else None
                if close_val is None or atr_val is None or pd.isna(atr_val):
                    continue

                close_f = float(close_val)
                atr_f   = float(atr_val)

                # ── Decision trace ───────────────────────────────────────────
                stop_price   = round(close_f - atr_f, 2)
                target_price = round(close_f + p["sell_atr_mult"] * atr_f, 2)

                trace = build_decision_trace(
                    symbol        = symbol,
                    regime        = regime,
                    setup_type    = setup_type,
                    confluence    = confluence,
                    quality_score = quality_score,
                    volume_ratio  = volume_ratio,
                    entry_price   = close_f,
                    stop_price    = stop_price,
                    target_price  = target_price,
                    extra_factors = {
                        "Probability":  f"{prob:.1%}",
                        "Market Phase": phase,
                        "ATR":          f"${atr_f:.2f}",
                    },
                )

                signals.append({
                    "instrument_id":    instrument_id,
                    "date":             dt.date() if hasattr(dt, "date") else dt,
                    "probability":      round(prob, 4),
                    "confidence_tier":  tier,
                    "market_phase":     phase,
                    "buy_price":        round(close_f, 2),
                    "accumulate_price": round(close_f - p["accumulate_atr_mult"] * atr_f, 2),
                    "scale_price":      round(close_f + p["scale_atr_mult"] * atr_f, 2),
                    "sell_price":       target_price,
                    # ── Regime fields ─────────────────────────────────────────
                    "market_state":         regime,
                    "strategy_mode":        rule.strategy_type,
                    "setup_quality_score":  round(quality_score, 1),
                    "decision_trace":       trace[:4000],  # enforce DB limit
                })

            return signals
        finally:
            session.close()

    # ── generate_and_store ───────────────────────────────────────────────────

    def generate_and_store(self, symbol: str | None = None):
        """Generate and store regime-gated signals for one or all instruments."""
        session = get_session()
        try:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol)
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Generating signals"):
                sigs = self.generate_for_symbol(inst.symbol)
                if sigs:
                    self._store_signals(session, sigs)
                    total += len(sigs)

            logger.info(f"Generated {total} total signals (regime-gated)")
            return total
        finally:
            session.close()

    # ── get_latest_signals ───────────────────────────────────────────────────

    def get_latest_signals(
        self,
        min_probability: float = 0.0,
        tier:            str | None = None,
    ) -> list[dict]:
        """Return the most recent regime-approved signal per instrument."""
        session = get_session()
        try:
            instruments = session.execute(
                select(Instrument).order_by(Instrument.symbol)
            ).scalars().all()

            latest = []
            for inst in instruments:
                q = (
                    select(Signal)
                    .where(Signal.instrument_id == inst.id)
                    .where(Signal.probability >= min_probability)
                )
                if tier:
                    q = q.where(Signal.confidence_tier == tier.upper())
                q = q.order_by(Signal.date.desc()).limit(1)

                sig = session.execute(q).scalar()
                if sig:
                    latest.append({
                        "symbol":               inst.symbol,
                        "date":                 str(sig.date),
                        "probability":          sig.probability,
                        "confidence_tier":      sig.confidence_tier,
                        "market_phase":         sig.market_phase,
                        "market_state":         sig.market_state,
                        "strategy_mode":        sig.strategy_mode,
                        "setup_quality_score":  sig.setup_quality_score,
                        "buy_price":            sig.buy_price,
                        "accumulate_price":     sig.accumulate_price,
                        "scale_price":          sig.scale_price,
                        "sell_price":           sig.sell_price,
                        "decision_trace":       sig.decision_trace,
                    })

            return sorted(latest, key=lambda x: x["probability"], reverse=True)
        finally:
            session.close()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _store_signals(self, session, signals: list[dict]):
        update_cols = [
            "probability", "confidence_tier", "market_phase",
            "buy_price", "accumulate_price", "scale_price", "sell_price",
            "market_state", "strategy_mode", "setup_quality_score", "decision_trace",
        ]
        for i in range(0, len(signals), 500):
            batch = signals[i:i + 500]
            stmt  = pg_insert(Signal).values(batch)
            stmt  = stmt.on_conflict_do_update(
                index_elements=["instrument_id", "date"],
                set_={col: stmt.excluded[col] for col in update_cols},
            )
            session.execute(stmt)
        session.commit()

    @staticmethod
    def _get_market_state(session, instrument_id: int) -> str:
        """Fetch the most recent market state for an instrument from the DB."""
        from database.models import MarketState
        row = session.execute(
            select(MarketState.state)
            .where(MarketState.instrument_id == instrument_id)
            .order_by(desc(MarketState.date))
            .limit(1)
        ).scalar_one_or_none()
        return row or "COMPRESSION"

    @staticmethod
    def _get_confluence_score(session, instrument_id: int, as_of) -> float:
        """Fetch the most recent confluence score on or before *as_of*."""
        from database.models import ConfluenceScore
        row = session.execute(
            select(ConfluenceScore.confluence_score)
            .where(
                ConfluenceScore.instrument_id == instrument_id,
                ConfluenceScore.date <= as_of,
            )
            .order_by(desc(ConfluenceScore.date))
            .limit(1)
        ).scalar_one_or_none()
        return float(row) if row is not None else 0.0

    @staticmethod
    def _get_setup_quality_score(symbol: str) -> float:
        """Compute (or fetch cached) SetupQualityScore for the symbol."""
        try:
            from analytics_engine.setup_quality import SetupQualityScorer
            result = SetupQualityScorer().score_symbol(symbol)
            return float(result.get("quality_score", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _infer_setup_type(feat_row: dict, phase: str) -> str:
        """
        Heuristic: infer the most likely setup type from feature values.
        Used when the full confluence engine is not re-run per signal.
        """
        bb_pos    = float(feat_row.get("bb_position", 0.5) or 0.5)
        trend_str = float(feat_row.get("trend_strength", 0.0) or 0.0)
        vol_ratio = float(feat_row.get("volume_ratio", 1.0) or 1.0)

        if bb_pos >= 0.85 and vol_ratio >= 1.4:
            return "compression_breakout"
        if trend_str > 0.02 and bb_pos < 0.5:
            return "trend_pullback"
        if bb_pos <= 0.15:
            return "pattern_reversal"
        return "compression_breakout"  # default
