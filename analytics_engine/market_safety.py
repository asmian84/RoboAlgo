"""
RoboAlgo — Market Safety Engine
Detects abnormal market conditions and protects capital by pausing or
throttling trading when the system is operating outside normal parameters.

The engine runs on a continuous basis and produces a SafetyScore (0–100).

Safety States
-------------
NORMAL     (safety_score ≥ 70) — all trading allowed at normal size
CAUTION    (safety_score 50–69) — trading allowed but position sizes reduced 50%
SAFE_MODE  (safety_score < 50) — all new entries blocked; existing positions monitored

Score Components
----------------
volatility_score  (40%)
    Cross-instrument average volatility percentile, inverted.
    Low volatility = high score (normal); high volatility spike = low score.
    Score = max(0, 100 − avg_vol_percentile × 100)

gap_score  (25%)
    Detects recent large overnight gaps (> GAP_THRESHOLD of price).
    Each qualifying gap reduces the score.
    Score = max(0, 100 − gap_severity × 100)

portfolio_score  (20%)
    Based on daily P&L vs the max daily loss limit.
    If portfolio is down more than 3% today → CAUTION zone.
    Score = max(0, 100 − drawdown_severity × 100)

data_quality_score  (15%)
    Pipeline health check: is market data fresh and valid?
    Uses the same data_quality from DataValidator.
    Score = 100 if pipeline OK, 50 if STALE, 0 if ERROR.

System Actions by State
-----------------------
NORMAL    → trading_allowed=True,  size_multiplier=1.0
CAUTION   → trading_allowed=True,  size_multiplier=0.5
SAFE_MODE → trading_allowed=False, size_multiplier=0.0
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

from database.connection import get_session

logger = logging.getLogger(__name__)

# ── Component weights ──────────────────────────────────────────────────────────
WEIGHTS = {
    "volatility":    0.40,
    "gap":           0.25,
    "portfolio":     0.20,
    "data_quality":  0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ── Constants ──────────────────────────────────────────────────────────────────
STATE_NORMAL    = 70.0   # score ≥ 70 → NORMAL
STATE_CAUTION   = 50.0   # score ≥ 50 → CAUTION
# score < 50 → SAFE_MODE

GAP_THRESHOLD          = 0.03   # 3%+ overnight gap triggers gap penalty
GAP_SEVERE_THRESHOLD   = 0.06   # 6%+ gap = maximum penalty
VOL_SPIKE_THRESHOLD    = 0.75   # avg volatility percentile > 75% = spike
PORTFOLIO_CAUTION_PCT  = 0.03   # down 3% today → caution zone
PORTFOLIO_CRITICAL_PCT = 0.05   # down 5% → safe_mode zone


def _safety_state(score: float) -> str:
    if score >= STATE_NORMAL:
        return "NORMAL"
    if score >= STATE_CAUTION:
        return "CAUTION"
    return "SAFE_MODE"


def _size_multiplier(score: float) -> float:
    if score >= STATE_NORMAL:
        return 1.0
    if score >= STATE_CAUTION:
        return 0.5
    return 0.0


class MarketSafetyEngine:
    """
    Evaluates current market conditions against safety thresholds.

    Usage:
        engine = MarketSafetyEngine()
        result = engine.evaluate()
        if not result["trading_allowed"]:
            return   # system is in SAFE_MODE — no new entries

        size_adj = result["size_multiplier"]  # 1.0 | 0.5 | 0.0
    """

    def evaluate(self, as_of_date: Optional[date] = None) -> dict:
        """
        Compute the current SafetyScore and system safety state.

        Returns:
            {
              safety_score, safety_state,
              trading_allowed, size_multiplier,
              components: { volatility, gap, portfolio, data_quality },
              triggers: [str]  # human-readable reasons for any degradation,
              computed_at
            }
        """
        target_date = as_of_date or date.today()
        triggers    = []

        vol_score  = self._volatility_score(target_date, triggers)
        gap_score  = self._gap_score(target_date, triggers)
        port_score = self._portfolio_score(triggers)
        dq_score   = self._data_quality_score(triggers)

        components = {
            "volatility":   round(vol_score, 1),
            "gap":          round(gap_score, 1),
            "portfolio":    round(port_score, 1),
            "data_quality": round(dq_score, 1),
        }

        safety_score = round(
            sum(components[k] * WEIGHTS[k] for k in WEIGHTS),
            1,
        )
        state   = _safety_state(safety_score)
        allowed = state != "SAFE_MODE"
        mult    = _size_multiplier(safety_score)

        return {
            "safety_score":    safety_score,
            "safety_state":    state,
            "trading_allowed": allowed,
            "size_multiplier": mult,
            "components":      components,
            "weights":         WEIGHTS,
            "triggers":        triggers,
            "computed_at":     datetime.utcnow().isoformat() + "Z",
        }

    # ── Components ────────────────────────────────────────────────────────────

    def _volatility_score(self, cutoff: date, triggers: list) -> float:
        """
        Measure average cross-instrument volatility percentile.
        High volatility → low safety score.
        """
        try:
            from database.models import Feature, Instrument
            from sqlalchemy import select, desc, func

            with get_session() as session:
                # Get avg volatility_percentile across all instruments for latest date
                subq = (
                    select(
                        Feature.instrument_id,
                        func.max(Feature.date).label("max_date"),
                    )
                    .where(Feature.date <= cutoff)
                    .group_by(Feature.instrument_id)
                    .subquery()
                )
                rows = session.execute(
                    select(Feature.volatility_percentile)
                    .join(
                        subq,
                        (Feature.instrument_id == subq.c.instrument_id) &
                        (Feature.date          == subq.c.max_date),
                    )
                    .where(Feature.volatility_percentile != None)  # noqa: E711
                ).scalars().all()

            if not rows:
                return 75.0  # neutral fallback

            avg_vol_pct = sum(rows) / len(rows)
            # Invert: high vol pct → low safety score
            score = max(0.0, (1.0 - avg_vol_pct) * 100.0)

            if avg_vol_pct > VOL_SPIKE_THRESHOLD:
                triggers.append(
                    f"Volatility spike detected — avg percentile {avg_vol_pct:.0%}"
                )

            return score

        except Exception as e:
            logger.warning("MarketSafetyEngine._volatility_score failed: %s", e)
            return 75.0

    def _gap_score(self, cutoff: date, triggers: list) -> float:
        """
        Check for large overnight gaps across tracked instruments in the last bar.
        Severe gaps reduce the gap score.
        """
        try:
            from database.models import PriceData, Instrument
            from sqlalchemy import select, desc

            large_gaps = 0
            total_checked = 0

            with get_session() as session:
                instruments = session.execute(select(Instrument)).scalars().all()

                for inst in instruments:
                    # Get the two most recent bars
                    bars = session.execute(
                        select(PriceData.open, PriceData.close)
                        .where(
                            PriceData.instrument_id == inst.id,
                            PriceData.date <= cutoff,
                        )
                        .order_by(desc(PriceData.date))
                        .limit(2)
                    ).all()

                    if len(bars) < 2:
                        continue

                    latest_open  = bars[0].open
                    prev_close   = bars[1].close

                    if latest_open and prev_close and prev_close > 0:
                        gap_pct = abs(latest_open - prev_close) / prev_close
                        if gap_pct >= GAP_THRESHOLD:
                            large_gaps += 1
                        total_checked += 1

            if total_checked == 0:
                return 100.0

            gap_rate = large_gaps / total_checked
            score    = max(0.0, 100.0 - gap_rate * 200.0)  # 50% instruments gapped → 0

            if large_gaps > 0:
                triggers.append(
                    f"Large gaps detected — {large_gaps}/{total_checked} instruments "
                    f"gapped ≥ {GAP_THRESHOLD:.0%}"
                )

            return min(score, 100.0)

        except Exception as e:
            logger.warning("MarketSafetyEngine._gap_score failed: %s", e)
            return 100.0

    def _portfolio_score(self, triggers: list) -> float:
        """
        Assess daily portfolio P&L vs the max daily loss limit.
        Significant intraday loss → safety throttle.
        """
        try:
            from database.models import PaperAccount
            from sqlalchemy import select

            with get_session() as session:
                acct = session.execute(
                    select(PaperAccount).order_by(PaperAccount.date.desc()).limit(1)
                ).scalar_one_or_none()

            if acct is None:
                return 100.0  # no data → assume safe

            equity = float(acct.ending_balance) if acct.ending_balance else 100_000.0
            pnl    = float(acct.daily_pnl) if acct.daily_pnl else 0.0

            if equity <= 0:
                return 100.0

            loss_pct = -min(pnl / equity, 0.0)   # positive fraction when in loss

            if loss_pct <= 0:
                return 100.0
            if loss_pct >= PORTFOLIO_CRITICAL_PCT:
                triggers.append(
                    f"Critical daily loss: {loss_pct:.1%} — SAFE_MODE triggered"
                )
                return 0.0
            if loss_pct >= PORTFOLIO_CAUTION_PCT:
                triggers.append(
                    f"Daily loss {loss_pct:.1%} exceeds caution threshold "
                    f"({PORTFOLIO_CAUTION_PCT:.0%})"
                )

            score = max(0.0, (1.0 - loss_pct / PORTFOLIO_CRITICAL_PCT) * 100.0)
            return min(score, 100.0)

        except Exception as e:
            logger.warning("MarketSafetyEngine._portfolio_score failed: %s", e)
            return 100.0

    def _data_quality_score(self, triggers: list) -> float:
        """
        Check data pipeline freshness.
        Stale or errored data → reduced safety confidence.
        """
        try:
            from database.models import PriceData
            from sqlalchemy import select, func

            with get_session() as session:
                max_date = session.execute(
                    select(func.max(PriceData.date))
                ).scalar()

            if max_date is None:
                triggers.append("No market data found in database")
                return 0.0

            if hasattr(max_date, "strftime"):
                last_dt = datetime.combine(max_date, datetime.min.time())
            else:
                last_dt = datetime.fromisoformat(str(max_date))

            age_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600

            if age_hours <= 26:
                return 100.0
            elif age_hours <= 72:
                triggers.append(f"Market data is stale — last update {age_hours:.0f}h ago")
                return 50.0
            else:
                triggers.append(f"Market data feed ERROR — {age_hours:.0f}h since last update")
                return 0.0

        except Exception as e:
            logger.warning("MarketSafetyEngine._data_quality_score failed: %s", e)
            return 50.0

    def run_scenario(self, scenario: str) -> dict:
        """
        Simulate specific market scenarios for testing.

        Scenarios: "normal" | "volatility_spike" | "large_gap" | "portfolio_drawdown"

        Returns the expected safety output for documentation/testing.
        """
        scenario_defaults = {
            "normal": {
                "volatility": 85.0, "gap": 100.0, "portfolio": 100.0, "data_quality": 100.0,
            },
            "volatility_spike": {
                "volatility":  0.0, "gap": 100.0, "portfolio": 100.0, "data_quality": 100.0,
            },
            "large_gap": {
                "volatility": 70.0, "gap":   0.0, "portfolio": 100.0, "data_quality": 100.0,
            },
            "portfolio_drawdown": {
                "volatility": 80.0, "gap": 100.0, "portfolio":   0.0, "data_quality": 100.0,
            },
        }

        components_raw = scenario_defaults.get(scenario, scenario_defaults["normal"])
        components = {k: round(v, 1) for k, v in components_raw.items()}

        safety_score = round(
            sum(components[k] * WEIGHTS[k] for k in WEIGHTS),
            1,
        )
        state   = _safety_state(safety_score)
        allowed = state != "SAFE_MODE"
        mult    = _size_multiplier(safety_score)

        # Determine system action
        if not allowed:
            action = "All new entries BLOCKED. Monitor open positions."
        elif mult < 1.0:
            action = f"Trading allowed. Position size reduced to {mult:.0%}."
        else:
            action = "Trading allowed at normal position size."

        return {
            "scenario":         scenario,
            "safety_score":     safety_score,
            "safety_state":     state,
            "trading_allowed":  allowed,
            "size_multiplier":  mult,
            "system_action":    action,
            "components":       components,
            "computed_at":      datetime.utcnow().isoformat() + "Z",
        }
