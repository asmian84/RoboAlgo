"""
RoboAlgo — Regime-Adaptive Strategy Engine
Dynamically adjusts trading behavior based on the current market state.

State → Strategy Profile mapping:
  COMPRESSION  → watchlist only, no entries, identify setups only
  TREND        → trend continuation trades, pullback entries, standard risk
  EXPANSION    → breakout trades, larger positions, aggressive entries
  CHAOS        → defensive mode, half size, require higher confluence (≥85)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyProfile:
    """Complete strategy configuration for a given market state."""
    state: str
    strategy_type: str
    position_multiplier: float
    confluence_threshold: float
    max_positions: int
    risk_per_trade: float
    allowed_setups: list[str]
    entry_notes: str
    size_description: str


# ── Strategy Profiles per Regime ──────────────────────────────────────────────
REGIME_PROFILES = {
    "COMPRESSION": StrategyProfile(
        state                = "COMPRESSION",
        strategy_type        = "Watchlist Builder",
        position_multiplier  = 0.0,          # no new trades
        confluence_threshold = 999.0,         # effectively blocked
        max_positions        = 0,
        risk_per_trade       = 0.0,
        allowed_setups       = [],            # setups only, no entries
        entry_notes          = "Compression detected. Building watchlist only. Wait for breakout trigger.",
        size_description     = "NO TRADES — compression phase, waiting for expansion",
    ),
    "TREND": StrategyProfile(
        state                = "TREND",
        strategy_type        = "Trend Continuation",
        position_multiplier  = 1.0,
        confluence_threshold = 65.0,
        max_positions        = 5,
        risk_per_trade       = 0.015,         # 1.5% risk
        allowed_setups       = ["trend_pullback", "compression_breakout", "liquidity_sweep"],
        entry_notes          = "Trend confirmed. Enter on pullbacks to MA20/MA50. Volume confirmation required.",
        size_description     = "Standard sizing (1.0×)",
    ),
    "EXPANSION": StrategyProfile(
        state                = "EXPANSION",
        strategy_type        = "Volatility Breakout",
        position_multiplier  = 1.25,          # +25% size
        confluence_threshold = 60.0,
        max_positions        = 6,
        risk_per_trade       = 0.02,           # 2% risk
        allowed_setups       = [
            "compression_breakout", "trend_pullback",
            "pattern_reversal", "wyckoff_spring", "liquidity_sweep",
        ],
        entry_notes          = "Expansion breakout. Enter aggressively on volume confirmation. Wider stops.",
        size_description     = "Increased sizing (1.25×) — expansion regime",
    ),
    "CHAOS": StrategyProfile(
        state                = "CHAOS",
        strategy_type        = "Defensive Mode",
        position_multiplier  = 0.5,           # -50% size
        confluence_threshold = 85.0,          # require very high conviction
        max_positions        = 2,
        risk_per_trade       = 0.005,          # 0.5% risk
        allowed_setups       = ["compression_breakout", "pattern_reversal"],
        entry_notes          = "Chaos regime. Defensive mode. Only highest conviction setups (≥85 confluence).",
        size_description     = "Reduced sizing (0.5×) — chaos regime",
    ),
}


class RegimeAdaptiveEngine:
    """
    Sits between Market State Engine and Signal Engine.
    Determines whether and how to execute signals based on current regime.

    Usage:
        engine = RegimeAdaptiveEngine()
        profile = engine.get_profile("SOXL")       # auto-fetches from DB
        profile = engine.get_profile_for_state("EXPANSION")  # direct lookup

        decision = engine.validate_signal(
            symbol="SOXL",
            market_state="EXPANSION",
            confluence_score=84,
            setup_type="compression_breakout",
        )
    """

    def get_profile(self, symbol: str) -> StrategyProfile:
        """
        Get the strategy profile for a symbol by querying its current market state.
        Falls back to COMPRESSION profile if no state available.
        """
        state = self._get_market_state(symbol)
        return REGIME_PROFILES.get(state, REGIME_PROFILES["COMPRESSION"])

    def get_profile_for_state(self, state: str) -> StrategyProfile:
        """Get strategy profile directly from a state string."""
        return REGIME_PROFILES.get(state.upper(), REGIME_PROFILES["COMPRESSION"])

    def validate_signal(
        self,
        symbol: str,
        market_state: str,
        confluence_score: float,
        setup_type: str,
        base_position_size: int = 0,
        account_equity: float = 0.0,
    ) -> dict:
        """
        Validate a signal against the current regime strategy profile.
        Returns approval decision and adjusted position size.
        """
        profile = self.get_profile_for_state(market_state)

        reasons = []
        warnings = []

        # ── Check 1: State allows trading at all ───────────────────────────────
        if profile.position_multiplier == 0.0:
            return {
                "approved":       False,
                "downgraded":     True,
                "tier_override":  "WATCH",
                "signal_state":   "SETUP",  # lifecycle state
                "profile":        self._profile_to_dict(profile),
                "reason":         f"{market_state} state — signal downgraded to WATCH (watchlist only)",
                "warnings":       warnings,
                "adjusted_size":  0,
            }

        # ── Check 2: Confluence threshold ──────────────────────────────────────
        if confluence_score < profile.confluence_threshold:
            return {
                "approved":      False,
                "downgraded":    True,
                "tier_override": "WATCH",
                "signal_state":  "TRIGGER",
                "profile":       self._profile_to_dict(profile),
                "reason": (
                    f"Confluence {confluence_score:.1f} < "
                    f"{profile.confluence_threshold:.0f} required for {market_state}"
                ),
                "warnings": warnings,
                "adjusted_size": 0,
            }

        # ── Check 3: Setup type allowed ────────────────────────────────────────
        if profile.allowed_setups and setup_type not in profile.allowed_setups:
            return {
                "approved":      False,
                "downgraded":    True,
                "tier_override": "WATCH",
                "signal_state":  "TRIGGER",
                "profile":       self._profile_to_dict(profile),
                "reason":        f"Setup '{setup_type}' not allowed in {market_state} regime",
                "warnings":      warnings,
                "adjusted_size": 0,
            }

        # ── All checks passed — compute adjusted size ─────────────────────────
        adjusted_size = int(base_position_size * profile.position_multiplier)
        if adjusted_size != base_position_size:
            warnings.append(
                f"Position size adjusted: {base_position_size} → {adjusted_size} "
                f"({profile.position_multiplier:.2f}× {market_state} multiplier)"
            )

        return {
            "approved":           True,
            "downgraded":         False,
            "tier_override":      None,
            "signal_state":       "ENTRY",
            "profile":            self._profile_to_dict(profile),
            "adjusted_size":      adjusted_size,
            "position_multiplier": profile.position_multiplier,
            "risk_per_trade":     profile.risk_per_trade,
            "strategy_type":      profile.strategy_type,
            "entry_notes":        profile.entry_notes,
            "warnings":           warnings,
        }

    def get_opportunity_map(self) -> dict:
        """
        Group all primary watchlist instruments by their current regime.
        Returns a dict: {state → [symbols]}
        """
        from config.settings import PRIMARY_WATCHLIST
        from market_state_engine.state import MarketStateEngine

        mse = MarketStateEngine()
        result = {"EXPANSION": [], "TREND": [], "COMPRESSION": [], "CHAOS": []}

        for sym in PRIMARY_WATCHLIST:
            state_data = mse.get_latest(sym)
            state = state_data.get("state", "COMPRESSION") if state_data else "COMPRESSION"
            result[state].append(sym)

        return result

    def get_regime_strategy_performance(self) -> list[dict]:
        """Return historical performance by regime+setup combination."""
        from database.models import RegimeStrategyPerformance
        from sqlalchemy import select, desc

        with get_session() as session:
            from database.connection import get_session
            with get_session() as s:
                rows = s.execute(
                    select(RegimeStrategyPerformance)
                    .order_by(desc(RegimeStrategyPerformance.expected_value))
                ).scalars().all()
                return [{
                    "market_state":   r.market_state,
                    "setup_type":     r.setup_type,
                    "win_rate":       r.win_rate,
                    "avg_return":     r.avg_return,
                    "expected_value": r.expected_value,
                    "trade_count":    r.trade_count,
                } for r in rows]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_market_state(self, symbol: str) -> str:
        """Fetch latest market state from DB."""
        from market_state_engine.state import MarketStateEngine
        engine = MarketStateEngine()
        result = engine.get_latest(symbol)
        return result.get("state", "COMPRESSION") if result else "COMPRESSION"

    def _profile_to_dict(self, profile: StrategyProfile) -> dict:
        return {
            "state":                profile.state,
            "strategy_type":        profile.strategy_type,
            "position_multiplier":  profile.position_multiplier,
            "confluence_threshold": profile.confluence_threshold,
            "max_positions":        profile.max_positions,
            "risk_per_trade":       profile.risk_per_trade,
            "allowed_setups":       profile.allowed_setups,
            "size_description":     profile.size_description,
            "entry_notes":          profile.entry_notes,
        }


def get_session():
    """Import helper to avoid circular imports."""
    from database.connection import get_session as _get_session
    return _get_session()
