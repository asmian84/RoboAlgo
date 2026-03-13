"""
RoboAlgo — Portfolio Engine
Manages risk, position sizing, and exposure limits across all active trades.

v2: All regime-dependent risk parameters are sourced from the Regime Playbook.
  Hardcoded constants are kept only for structural limits (sector, daily-loss cap).

Regime Risk Table (from playbook):
  EXPANSION    risk_per_trade=2.0%  max_positions=5
  TREND        risk_per_trade=1.5%  max_positions=5
  COMPRESSION  risk_per_trade=1.0%  max_positions=3  (range/hedge only)
  CHAOS        risk_per_trade=0.5%  max_positions=2  (quality ≥ 75 required)

Structural limits (regime-independent):
  max_sector_exposure = 40%
  max_daily_loss      = 5%
  max_position_exposure = 25% of equity

Position sizing formula:
  position_size = (account_equity × risk_per_trade × position_multiplier) / stop_distance
"""

import logging
from typing import Optional

from database.connection import get_session
from database.models import PaperPosition, PaperAccount
from sqlalchemy import select, func

logger = logging.getLogger(__name__)

# ── Structural limits (never regime-dependent) ─────────────────────────────────
MAX_SECTOR_EXPOSURE   = 0.40   # 40% max in single sector
MAX_DAILY_LOSS        = 0.05   # 5% max daily drawdown
MAX_POSITION_EXPOSURE = 0.25   # single position max 25% of equity

# Sector mapping for leveraged ETF universe
SECTOR_MAP = {
    "SOXL": "semiconductor", "SOXS": "semiconductor",
    "TQQQ": "nasdaq",        "SQQQ": "nasdaq",
    "UPRO": "sp500",         "SPXU": "sp500",
    "TNA":  "small_cap",     "TZA":  "small_cap",
    "TECL": "technology",    "TECS": "technology",
    "FAS":  "financial",     "FAZ":  "financial",
    "LABU": "biotech",       "LABD": "biotech",
    "GUSH": "energy",        "DRIP": "energy",
    "BOIL": "energy",        "KOLD": "energy",
    "NVDL": "single_stock",  "NVDS": "single_stock",
    "TSLL": "single_stock",  "TSLZ": "single_stock",
    "MSTU": "single_stock",  "MSTZ": "single_stock",
}


class PortfolioManager:
    """
    Validates and sizes new trades against portfolio + regime rules.

    All regime-specific risk_per_trade and max_positions values come
    directly from the Regime Playbook — no hardcoded regime constants here.

    Usage:
        pm = PortfolioManager()
        result = pm.validate_trade(
            "SOXL", entry=52.30, stop=49.80,
            market_state="EXPANSION",
            setup_quality_score=82,
        )
        shares, reasons = result["shares"], result["reasons"]
    """

    def validate_trade(
        self,
        symbol:              str,
        entry_price:         float,
        stop_price:          float,
        account_equity:      Optional[float] = None,
        market_state:        str   = "TREND",
        confluence_score:    float = 0.0,
        setup_quality_score: float = 0.0,
    ) -> dict:
        """
        Validate and size a proposed trade.

        Returns: {approved, shares, position_value, risk_dollars, reasons, warnings,
                  regime_rule, risk_per_trade, max_positions}
        """
        from strategy_engine.regime_playbook import get_rule

        if account_equity is None:
            account_equity = self._get_account_equity()

        rule     = get_rule(market_state)
        reasons  = []
        warnings = []

        # ── Check 1: Regime allows directional entries ─────────────────────────
        if rule.position_multiplier == 0.0:
            return {
                "approved": False,
                "shares":   0,
                "reason":   f"{market_state} — {rule.entry_description}",
                "reasons":  [f"Market state: {market_state} — {rule.size_description}"],
                "regime_rule": rule.strategy_type,
            }

        # ── Check 2: SetupQualityScore gate ───────────────────────────────────
        if rule.quality_score_min > 0 and setup_quality_score < rule.quality_score_min:
            return {
                "approved": False,
                "shares":   0,
                "reasons": [
                    f"SetupQualityScore {setup_quality_score:.0f} < "
                    f"{rule.quality_score_min:.0f} required for {market_state} "
                    f"({rule.strategy_type})"
                ],
                "regime_rule": rule.strategy_type,
            }

        # ── Check 3: Per-regime max positions ─────────────────────────────────
        open_positions = self._count_open_positions()
        if open_positions >= rule.max_positions:
            return {
                "approved": False,
                "shares":   0,
                "reasons": [
                    f"Max positions for {market_state} reached "
                    f"({open_positions}/{rule.max_positions})"
                ],
                "regime_rule": rule.strategy_type,
            }

        # ── Check 4: Daily Loss Limit ─────────────────────────────────────────
        daily_pnl_pct = self._get_daily_pnl_pct(account_equity)
        if daily_pnl_pct <= -MAX_DAILY_LOSS:
            return {
                "approved": False,
                "shares":   0,
                "reasons": [
                    f"Daily loss limit hit ({daily_pnl_pct:.1%} ≤ -{MAX_DAILY_LOSS:.0%})"
                ],
            }

        # ── Check 5: Sector Exposure ──────────────────────────────────────────
        sector          = SECTOR_MAP.get(symbol.upper(), "other")
        sector_exposure = self._get_sector_exposure(sector, account_equity)
        if sector_exposure >= MAX_SECTOR_EXPOSURE:
            warnings.append(
                f"Sector '{sector}' exposure {sector_exposure:.1%} near limit "
                f"{MAX_SECTOR_EXPOSURE:.0%}"
            )

        # ── Position Sizing ───────────────────────────────────────────────────
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return {
                "approved": False,
                "shares":   0,
                "reasons":  ["Invalid stop distance (stop = entry)"],
            }

        # Per-regime risk from playbook (no hardcoded constants)
        risk_dollars = account_equity * rule.risk_per_trade * rule.position_multiplier
        warnings.append(
            f"{market_state} ({rule.strategy_type}): "
            f"{rule.risk_per_trade:.1%} risk × {rule.position_multiplier:.2f}× size"
        )

        # Confluence bonus: high conviction adds +10%, capped at 4% equity risk
        if confluence_score >= 85:
            risk_dollars = min(risk_dollars * 1.1, account_equity * 0.04)
            warnings.append("High confluence (≥85): +10% size bonus applied")

        # Core sizing formula
        shares = int(risk_dollars / stop_distance)

        # Cap at max position exposure
        max_position_value = account_equity * MAX_POSITION_EXPOSURE
        if shares * entry_price > max_position_value:
            shares = int(max_position_value / entry_price)
            warnings.append(f"Capped at {MAX_POSITION_EXPOSURE:.0%} max position exposure")

        # Sector cap
        if sector_exposure + (shares * entry_price / account_equity) > MAX_SECTOR_EXPOSURE:
            max_add = (MAX_SECTOR_EXPOSURE - sector_exposure) * account_equity
            shares  = min(shares, int(max_add / entry_price))
            warnings.append(
                f"Sector cap applied — reduced to respect {MAX_SECTOR_EXPOSURE:.0%} limit"
            )

        if shares <= 0:
            return {
                "approved": False,
                "shares":   0,
                "reasons":  ["Position size calculated as 0 after all constraints"],
                "warnings": warnings,
            }

        position_value = shares * entry_price
        actual_risk    = shares * stop_distance

        return {
            "approved":          True,
            "shares":            shares,
            "position_value":    round(position_value, 2),
            "risk_dollars":      round(actual_risk, 2),
            "risk_pct":          round(actual_risk / account_equity, 4),
            "position_pct":      round(position_value / account_equity, 4),
            "state_multiplier":  rule.position_multiplier,
            "risk_per_trade":    rule.risk_per_trade,
            "max_positions":     rule.max_positions,
            "regime_rule":       rule.strategy_type,
            "sector":            sector,
            "sector_exposure":   round(sector_exposure, 4),
            "open_positions":    open_positions,
            "reasons":           reasons,
            "warnings":          warnings,
        }

    def get_portfolio_summary(self) -> dict:
        """Return current portfolio state and risk metrics."""
        equity         = self._get_account_equity()
        open_positions = self._count_open_positions()
        daily_pnl_pct  = self._get_daily_pnl_pct(equity)

        sectors = {}
        with get_session() as session:
            positions = session.execute(select(PaperPosition)).scalars().all()
            for pos in positions:
                sector = SECTOR_MAP.get(pos.symbol, "other")
                val    = pos.position_value or 0
                sectors[sector] = sectors.get(sector, 0) + val

        sector_pcts = {s: round(v / max(equity, 1), 4) for s, v in sectors.items()}

        return {
            "account_equity":        round(equity, 2),
            "open_positions":        open_positions,
            "max_positions":         5,
            "daily_pnl_pct":         round(daily_pnl_pct, 4),
            "daily_loss_limit":      MAX_DAILY_LOSS,
            "sector_exposure":       sector_pcts,
            "slots_available":       max(0, 5 - open_positions),
            "risk_budget_remaining": round(max(0, MAX_DAILY_LOSS + daily_pnl_pct), 4),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_account_equity(self) -> float:
        with get_session() as session:
            acct = session.execute(
                select(PaperAccount).order_by(PaperAccount.date.desc()).limit(1)
            ).scalar_one_or_none()
            return float(acct.ending_balance) if acct else 100_000.0

    def _count_open_positions(self) -> int:
        with get_session() as session:
            result = session.execute(
                select(func.count(PaperPosition.id))
            ).scalar_one()
            return int(result or 0)

    def _get_daily_pnl_pct(self, equity: float) -> float:
        with get_session() as session:
            today_acct = session.execute(
                select(PaperAccount).order_by(PaperAccount.date.desc()).limit(1)
            ).scalar_one_or_none()
            if not today_acct or equity <= 0:
                return 0.0
            return float(today_acct.daily_pnl or 0) / equity

    def _get_sector_exposure(self, sector: str, equity: float) -> float:
        if equity <= 0:
            return 0.0
        with get_session() as session:
            positions = session.execute(select(PaperPosition)).scalars().all()
            sector_val = sum(
                (pos.position_value or 0)
                for pos in positions
                if SECTOR_MAP.get(pos.symbol, "other") == sector
            )
            return sector_val / equity
