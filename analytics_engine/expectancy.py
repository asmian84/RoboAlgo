"""
RoboAlgo — Expectancy Engine
Tracks real performance of each setup type to drive data-driven signal weighting.

EV Formula:
  EV = (win_rate × avg_win) − (loss_rate × avg_loss)

Setup Types:
  compression_breakout  → primary volatility expansion trade
  trend_pullback        → trend continuation after retracement
  liquidity_sweep       → entry after liquidity grab/sweep
  pattern_reversal      → harmonic or chart pattern completion
  wyckoff_spring        → spring event entry in accumulation

The confluence engine queries setup EV before outputting a signal.
Negative EV → downgrade to WATCH regardless of confluence score.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select, func, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import TradeLifecycle, SetupPerformance, RegimeStrategyPerformance

logger = logging.getLogger(__name__)

# Known setup types
SETUP_TYPES = [
    "compression_breakout",
    "trend_pullback",
    "liquidity_sweep",
    "pattern_reversal",
    "wyckoff_spring",
]

MARKET_STATES = ["COMPRESSION", "TREND", "EXPANSION", "CHAOS"]

# Minimum trades before EV is considered reliable
MIN_TRADES_RELIABLE = 10


class ExpectancyEngine:
    """
    Tracks and updates expected value metrics per setup type and market state.

    Usage:
        engine = ExpectancyEngine()
        # After a trade closes:
        engine.record_trade_outcome(symbol="SOXL", setup_type="compression_breakout",
                                    market_state="EXPANSION", pnl=450.0,
                                    return_pct=0.08)
        # Query before entering a trade:
        ev = engine.get_setup_ev("compression_breakout", "EXPANSION")
        if ev < 0: downgrade_signal()
    """

    def record_trade_outcome(
        self,
        symbol: str,
        setup_type: str,
        market_state: str,
        pnl: float,
        return_pct: float,
        confluence_score: float = 0.0,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        position_size: int = 0,
        stop_price: float = 0.0,
        target_price: float = 0.0,
        holding_period: int = 0,
        exit_reason: str = "unknown",
        direction: str = "long",
    ):
        """Record a completed trade and update setup performance metrics."""
        with get_session() as session:
            # Record in lifecycle
            lifecycle = TradeLifecycle(
                symbol           = symbol,
                setup_type       = setup_type,
                market_state     = market_state,
                state            = "EXIT",
                entry_price      = entry_price,
                exit_price       = exit_price,
                stop_price       = stop_price,
                target_price     = target_price,
                position_size    = position_size,
                direction        = direction,
                exit_timestamp   = datetime.utcnow(),
                pnl              = pnl,
                return_percent   = return_pct,
                holding_period   = holding_period,
                exit_reason      = exit_reason,
                confluence_score = confluence_score,
            )
            session.add(lifecycle)
            session.commit()

            # Recalculate metrics
            self._recalculate_setup_performance(session, setup_type, market_state)
            self._recalculate_regime_performance(session, setup_type, market_state)

    def get_setup_ev(
        self, setup_type: str, market_state: Optional[str] = None
    ) -> float:
        """
        Return expected value for a setup type.
        Returns 0.0 if insufficient data (less than MIN_TRADES_RELIABLE).
        """
        with get_session() as session:
            query = select(SetupPerformance).where(
                SetupPerformance.setup_type == setup_type
            )
            if market_state:
                query = query.where(SetupPerformance.market_state == market_state)
            row = session.execute(query.limit(1)).scalar_one_or_none()

            if not row or (row.trade_count or 0) < MIN_TRADES_RELIABLE:
                return 0.0  # insufficient data — neutral
            return float(row.expected_value or 0)

    def get_all_setup_performance(self) -> list[dict]:
        """Return performance metrics for all setup types."""
        with get_session() as session:
            rows = session.execute(
                select(SetupPerformance)
                .order_by(desc(SetupPerformance.expected_value))
            ).scalars().all()
            return [self._perf_to_dict(r) for r in rows]

    def get_signal_weight_adjustment(
        self, setup_type: str, market_state: str
    ) -> float:
        """
        Return a multiplier (0–1.5) for adjusting signal weights based on EV.
        Negative EV → 0.0 (suppress)
        Zero EV     → 1.0 (neutral)
        High EV     → up to 1.5 (amplify)
        """
        ev = self.get_setup_ev(setup_type, market_state)
        if ev < 0:
            return 0.0   # suppress — negative expectancy
        if ev < 0.01:
            return 1.0   # neutral
        return float(np.clip(1.0 + ev * 5, 1.0, 1.5))  # boost for positive EV

    def classify_setup_type(
        self,
        is_compression: bool,
        is_breakout: bool,
        wyckoff_phase: Optional[str],
        has_chart_pattern: bool,
        has_harmonic: bool,
        above_vwap: bool,
        swept_level: bool,
    ) -> str:
        """
        Classify a trade into its setup type based on signal context.
        Priority order: breakout > spring > reversal > sweep > pullback.
        """
        if is_compression and is_breakout:
            return "compression_breakout"
        if wyckoff_phase == "Accumulation" and (
            wyckoff_phase and "spring" in str(wyckoff_phase).lower()
        ):
            return "wyckoff_spring"
        if has_harmonic or has_chart_pattern:
            return "pattern_reversal"
        if swept_level:
            return "liquidity_sweep"
        return "trend_pullback"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _recalculate_setup_performance(
        self, session, setup_type: str, market_state: str
    ):
        """Recompute and upsert SetupPerformance for a (setup_type, market_state) pair."""
        trades = session.execute(
            select(TradeLifecycle)
            .where(TradeLifecycle.setup_type == setup_type)
            .where(TradeLifecycle.market_state == market_state)
            .where(TradeLifecycle.state == "EXIT")
        ).scalars().all()

        if not trades:
            return

        returns = [float(t.return_percent or 0) for t in trades]
        pnls    = [float(t.pnl or 0) for t in trades]
        wins    = [r for r in returns if r > 0]
        losses  = [r for r in returns if r <= 0]

        win_rate       = len(wins) / max(len(returns), 1)
        avg_win        = float(np.mean(wins))   if wins   else 0.0
        avg_loss       = float(np.mean(losses)) if losses else 0.0
        gross_profit   = sum(p for p in pnls if p > 0)
        gross_loss     = abs(sum(p for p in pnls if p <= 0))
        profit_factor  = gross_profit / max(gross_loss, 1e-8)
        ev             = win_rate * avg_win + (1 - win_rate) * avg_loss

        record = {
            "setup_type":    setup_type,
            "market_state":  market_state,
            "win_rate":      round(win_rate, 4),
            "avg_win":       round(avg_win, 4),
            "avg_loss":      round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "expected_value": round(ev, 4),
            "trade_count":   len(trades),
            "updated_at":    datetime.utcnow(),
        }

        stmt = pg_insert(SetupPerformance).values([record])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_setup_perf_type_state",
            set_={k: stmt.excluded[k] for k in record if k not in ("setup_type", "market_state")}
        )
        session.execute(stmt)
        session.commit()

    def _recalculate_regime_performance(
        self, session, setup_type: str, market_state: str
    ):
        """Recompute RegimeStrategyPerformance for a regime+setup combination."""
        trades = session.execute(
            select(TradeLifecycle)
            .where(TradeLifecycle.market_state == market_state)
            .where(TradeLifecycle.setup_type == setup_type)
            .where(TradeLifecycle.state == "EXIT")
        ).scalars().all()

        if not trades:
            return

        returns = [float(t.return_percent or 0) for t in trades]
        wins    = [r for r in returns if r > 0]
        losses  = [r for r in returns if r <= 0]
        win_rate = len(wins) / max(len(returns), 1)
        avg_ret  = float(np.mean(returns)) if returns else 0.0
        ev       = win_rate * (float(np.mean(wins)) if wins else 0.0) + \
                   (1 - win_rate) * (float(np.mean(losses)) if losses else 0.0)

        record = {
            "market_state":  market_state,
            "setup_type":    setup_type,
            "win_rate":      round(win_rate, 4),
            "avg_return":    round(avg_ret, 4),
            "expected_value": round(ev, 4),
            "trade_count":   len(trades),
            "updated_at":    datetime.utcnow(),
        }

        stmt = pg_insert(RegimeStrategyPerformance).values([record])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_regime_strategy_perf",
            set_={k: stmt.excluded[k] for k in record if k not in ("market_state", "setup_type")}
        )
        session.execute(stmt)
        session.commit()

    def _perf_to_dict(self, row: SetupPerformance) -> dict:
        return {
            "setup_type":    row.setup_type,
            "market_state":  row.market_state,
            "win_rate":      row.win_rate,
            "avg_win":       row.avg_win,
            "avg_loss":      row.avg_loss,
            "profit_factor": row.profit_factor,
            "expected_value": row.expected_value,
            "trade_count":   row.trade_count,
            "is_reliable":   (row.trade_count or 0) >= MIN_TRADES_RELIABLE,
        }
