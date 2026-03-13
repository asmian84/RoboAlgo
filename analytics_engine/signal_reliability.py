"""
RoboAlgo — Signal Reliability Engine
Monitors whether each trading strategy setup type maintains statistical edge
over a rolling window of completed trades.

Automatically detects signal decay and downgrades or disables strategies
that have lost their edge, protecting capital before drawdowns compound.

Score: SignalReliabilityScore  0–100

Metrics (rolling window, default = 50 completed trades per setup type)
-----------------------------------------------------------------------
win_rate    — wins / total_trades
avg_win     — mean return of profitable trades (as a fraction)
avg_loss    — mean |return| of losing trades (as a fraction)
expectancy  — (win_rate × avg_win) − (loss_rate × avg_loss)
stability   — 1 − normalised std_dev of returns (high std = low stability)
drawdown    — maximum drawdown within the rolling window (cumulative equity)

Component Weights
-----------------
expectancy_score  40%
win_rate_score    30%
stability_score   20%
drawdown_score    10%

Status Thresholds
-----------------
≥ 70  →  "healthy"   — normal trading, full position size
50–69 →  "warning"   — reduce position size to 50%
< 50  →  "disabled"  — strategy temporarily suspended

Data Source
-----------
trade_lifecycle table, state = "EXIT", with return_percent populated.
Minimum 5 completed trades required to compute a meaningful score.

Storage: signal_reliability_scores table (one row per setup_type per date).
"""

import logging
import statistics
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy import select, desc, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import TradeLifecycle

logger = logging.getLogger(__name__)

# ── Component weights ──────────────────────────────────────────────────────────
WEIGHTS = {
    "expectancy":  0.40,
    "win_rate":    0.30,
    "stability":   0.20,
    "drawdown":    0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_SIZE       = 50      # rolling trade window per setup type
MIN_SAMPLE_SIZE   = 5       # minimum trades for a meaningful score
EXPECTANCY_FLOOR  = -0.10   # −10% avg return → score 0
EXPECTANCY_CAP    =  0.15   # +15% avg return → score 100
STD_DEV_CAP       =  0.20   # std dev ≥ 20% → stability score 0
DRAWDOWN_CAP      =  0.30   # drawdown ≥ 30% → drawdown score 0

# ── Status thresholds ─────────────────────────────────────────────────────────
STATUS_HEALTHY    = 70.0   # score ≥ 70 → full size
STATUS_WARNING    = 50.0   # score ≥ 50 → half size
# score < 50 → disabled

# ── Human-readable strategy names ─────────────────────────────────────────────
SETUP_TYPE_LABELS = {
    "compression_breakout": "Breakout",
    "trend_pullback":       "Pullback",
    "liquidity_sweep":      "Sweep",
    "pattern_reversal":     "Reversal",
    "wyckoff_spring":       "Wyckoff",
}


def _linear_score(value: float, floor: float, cap: float) -> float:
    """Map value linearly from 0 at floor to 100 at cap, clamped."""
    if value <= floor:
        return 0.0
    if value >= cap:
        return 100.0
    return (value - floor) / (cap - floor) * 100.0


def _max_drawdown(returns: list[float]) -> float:
    """
    Compute the maximum peak-to-trough drawdown from a sequence of trade returns.
    Returns a positive fraction (e.g. 0.15 = 15% drawdown).
    """
    if not returns:
        return 0.0
    equity = 1.0
    peak   = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / max(peak, 1e-9)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _reliability_status(score: float) -> str:
    if score >= STATUS_HEALTHY:
        return "healthy"
    if score >= STATUS_WARNING:
        return "warning"
    return "disabled"


def _position_multiplier(score: float) -> float:
    """Return position size multiplier based on reliability score."""
    if score >= STATUS_HEALTHY:
        return 1.0
    if score >= STATUS_WARNING:
        return 0.5
    return 0.0


class SignalReliabilityEngine:
    """
    Computes and tracks reliability scores for each trading setup type.

    A reliability score reflects whether a setup type is currently generating
    statistically significant positive expectancy over its recent trade history.

    Usage:
        engine = SignalReliabilityEngine()
        result = engine.compute_reliability("compression_breakout")
        score  = result["reliability_score"]        # 0–100
        mult   = result["position_multiplier"]      # 0.0 | 0.5 | 1.0
        status = result["status"]                   # "healthy" | "warning" | "disabled"

        # Fast gate call (checks DB first, falls back to live compute):
        gate   = engine.get_reliability_status("compression_breakout")
    """

    def compute_reliability(
        self,
        setup_type: str,
        window:     int = WINDOW_SIZE,
    ) -> dict:
        """
        Compute the Signal Reliability Score for *setup_type* from the last
        *window* completed trades.

        Returns:
            {
              setup_type, reliability_score, status, position_multiplier,
              metrics: { win_rate, avg_win, avg_loss, expectancy,
                         stability, max_drawdown, trade_count },
              component_scores: { expectancy, win_rate, stability, drawdown },
              weights, computed_at
            }
        """
        trades = self._fetch_completed_trades(setup_type, window)

        if len(trades) < MIN_SAMPLE_SIZE:
            return self._empty(
                setup_type,
                trade_count=len(trades),
                reason=f"Insufficient history ({len(trades)}/{MIN_SAMPLE_SIZE} trades)",
            )

        returns = [t["return_pct"] for t in trades]
        wins    = [r for r in returns if r > 0.0]
        losses  = [r for r in returns if r <= 0.0]

        win_rate  = len(wins) / len(returns)
        loss_rate = 1.0 - win_rate
        avg_win   = statistics.mean(wins)    if wins   else 0.0
        avg_loss  = abs(statistics.mean(losses)) if losses else 0.0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        std_dev  = statistics.stdev(returns) if len(returns) >= 2 else 0.0
        max_dd   = _max_drawdown(returns)

        # ── Component scores ──────────────────────────────────────────────────
        expectancy_score = _linear_score(expectancy, EXPECTANCY_FLOOR, EXPECTANCY_CAP)
        win_rate_score   = win_rate * 100.0
        stability_score  = max(0.0, (1.0 - std_dev / max(STD_DEV_CAP, 1e-9))) * 100.0
        stability_score  = min(stability_score, 100.0)
        drawdown_score   = max(0.0, (1.0 - max_dd / max(DRAWDOWN_CAP, 1e-9))) * 100.0
        drawdown_score   = min(drawdown_score, 100.0)

        component_scores = {
            "expectancy":  round(expectancy_score, 1),
            "win_rate":    round(win_rate_score,   1),
            "stability":   round(stability_score,  1),
            "drawdown":    round(drawdown_score,   1),
        }

        reliability_score = round(
            sum(component_scores[k] * WEIGHTS[k] for k in WEIGHTS),
            1,
        )
        status = _reliability_status(reliability_score)
        mult   = _position_multiplier(reliability_score)

        return {
            "setup_type":          setup_type,
            "strategy_label":      SETUP_TYPE_LABELS.get(setup_type, setup_type.replace("_", " ").title()),
            "reliability_score":   reliability_score,
            "status":              status,
            "position_multiplier": mult,
            "metrics": {
                "win_rate":    round(win_rate,   4),
                "avg_win":     round(avg_win,    4),
                "avg_loss":    round(avg_loss,   4),
                "expectancy":  round(expectancy, 4),
                "stability":   round(1.0 - std_dev / max(STD_DEV_CAP, 1e-9), 4),
                "max_drawdown":round(max_dd,     4),
                "trade_count": len(trades),
            },
            "component_scores": component_scores,
            "weights":          WEIGHTS,
            "computed_at":      datetime.utcnow().isoformat() + "Z",
        }

    def compute_all(self, window: int = WINDOW_SIZE) -> list[dict]:
        """
        Compute reliability for every setup type that has trade history.
        Returns results sorted by reliability_score descending.
        """
        setup_types = self._all_setup_types()
        results = []
        for st in setup_types:
            try:
                results.append(self.compute_reliability(st, window))
            except Exception as e:
                logger.warning("SignalReliabilityEngine failed for %s: %s", st, e)
        results.sort(key=lambda x: x["reliability_score"], reverse=True)
        return results

    def compute_and_store_all(self, as_of_date: Optional[date] = None) -> None:
        """Compute and persist SignalReliabilityScore for every setup type."""
        target_date = as_of_date or date.today()
        results = self.compute_all()
        for r in results:
            try:
                self._persist(r, target_date)
            except Exception as e:
                logger.warning(
                    "SignalReliabilityEngine persist failed for %s: %s",
                    r["setup_type"], e,
                )

    def get_reliability_status(self, setup_type: str) -> dict:
        """
        Fast gate call — tries the DB first (today's stored score), falls back
        to a live computation.  Always returns a result; never raises.

        Used by the strategy engine and signal engine before emitting signals.
        """
        try:
            stored = self._load_from_db(setup_type)
            if stored is not None:
                return stored
        except Exception:
            pass
        try:
            return self.compute_reliability(setup_type)
        except Exception as e:
            logger.warning("get_reliability_status fallback failed for %s: %s", setup_type, e)
            return self._empty(setup_type, reason=str(e))

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _fetch_completed_trades(
        self,
        setup_type: str,
        window:     int,
    ) -> list[dict]:
        """Fetch the last *window* EXIT trades for *setup_type*."""
        with get_session() as session:
            rows = session.execute(
                select(
                    TradeLifecycle.return_percent,
                    TradeLifecycle.pnl,
                    TradeLifecycle.exit_timestamp,
                )
                .where(
                    TradeLifecycle.setup_type    == setup_type,
                    TradeLifecycle.state         == "EXIT",
                    TradeLifecycle.return_percent != None,  # noqa: E711
                )
                .order_by(desc(TradeLifecycle.exit_timestamp))
                .limit(window)
            ).all()

        return [
            {
                "return_pct":   float(row.return_percent),
                "pnl":          float(row.pnl) if row.pnl is not None else 0.0,
                "exit_time":    str(row.exit_timestamp),
            }
            for row in rows
            if row.return_percent is not None
        ]

    def _all_setup_types(self) -> list[str]:
        """Return all distinct setup_types with at least one completed trade."""
        with get_session() as session:
            rows = session.execute(
                select(TradeLifecycle.setup_type)
                .where(
                    TradeLifecycle.state         == "EXIT",
                    TradeLifecycle.return_percent != None,  # noqa: E711
                    TradeLifecycle.setup_type    != None,   # noqa: E711
                )
                .distinct()
            ).scalars().all()
        return [r for r in rows if r]

    def _load_from_db(self, setup_type: str) -> Optional[dict]:
        """Return today's stored reliability record, or None if not found."""
        from database.models import SignalReliabilityScore
        today = date.today()
        with get_session() as session:
            row = session.execute(
                select(SignalReliabilityScore)
                .where(
                    SignalReliabilityScore.setup_type == setup_type,
                    SignalReliabilityScore.date       == today,
                )
                .limit(1)
            ).scalar_one_or_none()

        if row is None:
            return None

        return {
            "setup_type":          row.setup_type,
            "strategy_label":      SETUP_TYPE_LABELS.get(row.setup_type, row.setup_type),
            "reliability_score":   row.reliability_score,
            "status":              row.status,
            "position_multiplier": _position_multiplier(row.reliability_score),
            "metrics": {
                "win_rate":    row.win_rate,
                "avg_win":     row.avg_win,
                "avg_loss":    row.avg_loss,
                "expectancy":  row.expectancy,
                "max_drawdown":row.max_drawdown,
                "trade_count": row.trade_count,
            },
            "component_scores": {
                "expectancy": row.expectancy_score,
                "win_rate":   row.win_rate_score,
                "stability":  row.stability_score,
                "drawdown":   row.drawdown_score,
            },
            "weights":    WEIGHTS,
            "computed_at": row.computed_at.isoformat() + "Z" if row.computed_at else None,
        }

    def _persist(self, result: dict, score_date: date) -> None:
        from database.models import SignalReliabilityScore
        m = result.get("metrics", {})
        c = result.get("component_scores", {})

        record = {
            "setup_type":        result["setup_type"],
            "date":              score_date,
            "reliability_score": result["reliability_score"],
            "status":            result["status"],
            "win_rate":          m.get("win_rate"),
            "avg_win":           m.get("avg_win"),
            "avg_loss":          m.get("avg_loss"),
            "expectancy":        m.get("expectancy"),
            "max_drawdown":      m.get("max_drawdown"),
            "trade_count":       m.get("trade_count", 0),
            "expectancy_score":  c.get("expectancy"),
            "win_rate_score":    c.get("win_rate"),
            "stability_score":   c.get("stability"),
            "drawdown_score":    c.get("drawdown"),
            "computed_at":       datetime.utcnow(),
        }
        update_cols = [k for k in record if k != "setup_type"]

        with get_session() as session:
            stmt = pg_insert(SignalReliabilityScore).values([record])
            stmt = stmt.on_conflict_do_update(
                constraint="uq_signal_reliability_type_date",
                set_={c: stmt.excluded[c] for c in update_cols},
            )
            session.execute(stmt)
            session.commit()

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty(
        setup_type:  str,
        trade_count: int = 0,
        reason:      str = "",
    ) -> dict:
        """
        Return a neutral reliability record when no trade history exists.
        Neutral → full position size (don't penalise new / untested setups).
        """
        return {
            "setup_type":          setup_type,
            "strategy_label":      SETUP_TYPE_LABELS.get(setup_type, setup_type.replace("_", " ").title()),
            "reliability_score":   None,
            "status":              "no_data",
            "position_multiplier": 1.0,   # don't penalise when no history
            "metrics": {
                "win_rate":    None,
                "avg_win":     None,
                "avg_loss":    None,
                "expectancy":  None,
                "max_drawdown":None,
                "trade_count": trade_count,
            },
            "component_scores":    {},
            "weights":             WEIGHTS,
            "reason":              reason,
            "computed_at":         datetime.utcnow().isoformat() + "Z",
        }
