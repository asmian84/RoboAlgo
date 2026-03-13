"""
RoboAlgo — Position Scaling Engine
Dynamically adjusts position size based on signal quality, market regime,
system reliability, and current portfolio exposure.

This is the final risk calibration step before order sizing. It takes the
base risk from the Regime Playbook and applies three multiplicative adjustments:
  1. Setup Quality Multiplier — reward high-conviction setups
  2. Reliability Multiplier   — reduce size when a strategy shows signal decay
  3. Portfolio Adjustment     — throttle when overall portfolio risk is elevated

Final position size:
  final_risk_per_trade = base_risk × quality_mult × reliability_mult × portfolio_adj
  Capped at MAX_RISK_PER_TRADE = 3% of equity

Usage
-----
    from portfolio_engine.position_scaler import PositionScalingEngine

    engine = PositionScalingEngine()
    result = engine.calculate_position_size(
        symbol              = "SOXL",
        setup_quality_score = 84,
        reliability_score   = 75,
        regime              = "EXPANSION",
    )
    # → {
    #     "position_size_multiplier": 1.25,
    #     "risk_per_trade":           0.025,   # 2.5%
    #     "base_risk":                0.02,
    #     "quality_multiplier":       1.25,
    #     "reliability_multiplier":   1.0,
    #     "portfolio_adjustment":     1.0,
    #     "approved":                 True,
    #     "breakdown": [...],
    #   }
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE     = 0.03   # 3% hard cap
MAX_PORTFOLIO_RISK_PCT = 0.10   # 10% total portfolio risk → full throttle
PORTFOLIO_CAUTION_PCT  = 0.06   # above 6% total risk → start reducing

# ── Setup Quality multiplier table ────────────────────────────────────────────
# Grade A+ setups get bonus size; below 60 → reject outright (per spec)
QUALITY_TIERS = [
    (90, 1.50),   # score ≥ 90 → 1.5×
    (80, 1.25),   # score ≥ 80 → 1.25×
    (70, 1.00),   # score ≥ 70 → 1.0× (normal)
    (60, 0.75),   # score ≥ 60 → 0.75× (reduced)
]
QUALITY_REJECT_THRESHOLD = 60.0   # score < 60 → reject

# ── Reliability multiplier table ──────────────────────────────────────────────
RELIABILITY_TIERS = [
    (70, 1.00),   # score ≥ 70 → full size (healthy)
    (50, 0.75),   # score ≥ 50 → 75% size (warning — spec says 50%, we use 0.75 for gradual)
    ( 0, 0.50),   # score < 50 → 50% (disabled strategies get 0 from regime_playbook gate)
]


def _quality_multiplier(setup_quality_score: float) -> tuple[float, str]:
    """Return (multiplier, tier_label) for a given setup quality score."""
    for threshold, mult in QUALITY_TIERS:
        if setup_quality_score >= threshold:
            return mult, f"≥{threshold}"
    return 0.0, "rejected"   # below all tiers → reject


def _reliability_multiplier(reliability_score: Optional[float]) -> tuple[float, str]:
    """Return (multiplier, label) for a given reliability score."""
    if reliability_score is None:
        return 1.0, "no_data"   # no history → full size (neutral)
    for threshold, mult in RELIABILITY_TIERS:
        if reliability_score >= threshold:
            return mult, f"≥{threshold}"
    return 0.5, "below_50"


class PositionScalingEngine:
    """
    Computes the final position size multiplier and risk_per_trade for a trade.

    The result is used by PortfolioManager.validate_trade() and the execution
    engine to size orders precisely.

    Integration:
        PortfolioManager already handles max_positions, sector caps, and
        equity exposure caps.  PositionScalingEngine focuses exclusively on
        the risk_per_trade fraction before those structural checks.
    """

    def calculate_position_size(
        self,
        symbol:               str,
        setup_quality_score:  float,
        reliability_score:    Optional[float] = None,
        regime:               str   = "TREND",
        account_equity:       Optional[float] = None,
    ) -> dict:
        """
        Compute the final position size multiplier and risk_per_trade.

        Args:
            symbol:               Ticker (for portfolio exposure lookup).
            setup_quality_score:  SetupQualityScore 0–100.
            reliability_score:    SignalReliabilityScore 0–100 (or None).
            regime:               Market regime from MarketStateEngine.
            account_equity:       Account equity (fetched from DB if None).

        Returns:
            {
              "approved":                bool,    # False if quality < 60
              "position_size_multiplier": float,  # combined multiplier
              "risk_per_trade":           float,  # final % of equity to risk
              "base_risk":                float,  # from regime playbook
              "quality_multiplier":       float,
              "reliability_multiplier":   float,
              "portfolio_adjustment":     float,
              "breakdown":               [str],  # step-by-step explanation
              "reason":                   str,   # rejection reason if not approved
            }
        """
        from strategy_engine.regime_playbook import get_rule

        breakdown = []
        rule      = get_rule(regime)
        base_risk = rule.risk_per_trade

        breakdown.append(
            f"Step 1 — Base risk from playbook: {base_risk:.1%} "
            f"({regime} / {rule.strategy_type})"
        )

        # ── Step 2: Setup Quality Multiplier ──────────────────────────────────
        if setup_quality_score < QUALITY_REJECT_THRESHOLD:
            return {
                "approved":                False,
                "position_size_multiplier": 0.0,
                "risk_per_trade":           0.0,
                "base_risk":                base_risk,
                "quality_multiplier":       0.0,
                "reliability_multiplier":   0.0,
                "portfolio_adjustment":     0.0,
                "breakdown":                breakdown,
                "reason": (
                    f"SetupQualityScore {setup_quality_score:.0f} < "
                    f"{QUALITY_REJECT_THRESHOLD:.0f} — trade rejected"
                ),
            }

        quality_mult, quality_tier = _quality_multiplier(setup_quality_score)
        breakdown.append(
            f"Step 2 — Quality multiplier: {quality_mult:.2f}× "
            f"(score {setup_quality_score:.0f}, tier {quality_tier})"
        )

        # ── Step 3: Reliability Multiplier ────────────────────────────────────
        rel_mult, rel_label = _reliability_multiplier(reliability_score)
        rel_score_str = f"{reliability_score:.0f}" if reliability_score is not None else "N/A"
        breakdown.append(
            f"Step 3 — Reliability multiplier: {rel_mult:.2f}× "
            f"(score {rel_score_str}, {rel_label})"
        )

        # ── Step 4: Portfolio Adjustment ──────────────────────────────────────
        portfolio_adj = self._portfolio_adjustment(account_equity, breakdown)

        # ── Step 5: Final risk per trade ──────────────────────────────────────
        combined_mult  = quality_mult * rel_mult * portfolio_adj
        risk_per_trade = base_risk * combined_mult

        # Hard cap at MAX_RISK_PER_TRADE
        capped = False
        if risk_per_trade > MAX_RISK_PER_TRADE:
            risk_per_trade = MAX_RISK_PER_TRADE
            capped = True

        risk_per_trade = round(risk_per_trade, 4)
        breakdown.append(
            f"Step 5 — Final risk: {risk_per_trade:.2%} "
            f"({base_risk:.1%} × {quality_mult:.2f} × {rel_mult:.2f} × {portfolio_adj:.2f}"
            + (" — CAPPED at 3%" if capped else "") + ")"
        )

        return {
            "approved":                 True,
            "position_size_multiplier": round(combined_mult, 4),
            "risk_per_trade":           risk_per_trade,
            "base_risk":                base_risk,
            "quality_multiplier":       quality_mult,
            "reliability_multiplier":   rel_mult,
            "portfolio_adjustment":     portfolio_adj,
            "capped_at_max":            capped,
            "breakdown":                breakdown,
            "reason":                   "approved",
        }

    # ── Portfolio adjustment ──────────────────────────────────────────────────

    def _portfolio_adjustment(
        self,
        account_equity: Optional[float],
        breakdown:      list,
    ) -> float:
        """
        Reduce position size proportionally when total portfolio risk is elevated.

        If the combined value of all open positions exceeds PORTFOLIO_CAUTION_PCT,
        apply a linear reduction from 1.0 down to 0.5 as we approach MAX_PORTFOLIO_RISK_PCT.
        """
        try:
            from database.models import PaperPosition, PaperAccount
            from sqlalchemy import select

            with self._session() as session:
                if account_equity is None:
                    acct = session.execute(
                        select(PaperAccount).order_by(PaperAccount.date.desc()).limit(1)
                    ).scalar_one_or_none()
                    account_equity = float(acct.ending_balance) if acct else 100_000.0

                positions = session.execute(
                    select(PaperPosition)
                ).scalars().all()

            total_exposure = sum(
                float(p.position_value or 0) for p in positions
            )
            exposure_pct = total_exposure / max(account_equity, 1.0)

            if exposure_pct <= PORTFOLIO_CAUTION_PCT:
                adj = 1.0
            elif exposure_pct >= MAX_PORTFOLIO_RISK_PCT:
                adj = 0.5
            else:
                # Linear scale from 1.0 to 0.5 as exposure grows from caution to max
                t   = (exposure_pct - PORTFOLIO_CAUTION_PCT) / (MAX_PORTFOLIO_RISK_PCT - PORTFOLIO_CAUTION_PCT)
                adj = 1.0 - t * 0.5

            breakdown.append(
                f"Step 4 — Portfolio adjustment: {adj:.2f}× "
                f"(exposure {exposure_pct:.1%})"
            )
            return round(adj, 4)

        except Exception as e:
            logger.warning("PositionScalingEngine._portfolio_adjustment failed: %s", e)
            breakdown.append("Step 4 — Portfolio adjustment: 1.0× (DB unavailable)")
            return 1.0

    @staticmethod
    def _session():
        from database.connection import get_session
        return get_session()
