"""RoboAlgo — Strategy / Regime-Adaptive API Router
Endpoints for regime profiles, signal validation, and opportunity map.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


# ── Request Schemas ────────────────────────────────────────────────────────────

class ValidateSignalRequest(BaseModel):
    symbol:             str
    market_state:       str
    confluence_score:   float
    setup_type:         str
    base_position_size: int   = 0
    account_equity:     float = 0.0


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/profile/{symbol}")
def get_profile_for_symbol(symbol: str):
    """
    Get the regime-appropriate strategy profile for a symbol.
    Auto-fetches current market state from DB.
    """
    from strategy_engine.regime_adaptive import RegimeAdaptiveEngine
    engine  = RegimeAdaptiveEngine()
    profile = engine.get_profile(symbol.upper())
    return {"symbol": symbol.upper(), "profile": _profile_to_dict(profile)}


@router.get("/profile/state/{state}")
def get_profile_for_state(state: str):
    """Get strategy profile directly by market state name."""
    from strategy_engine.regime_adaptive import RegimeAdaptiveEngine
    engine  = RegimeAdaptiveEngine()
    profile = engine.get_profile_for_state(state.upper())
    return {"state": state.upper(), "profile": _profile_to_dict(profile)}


@router.post("/validate-signal")
def validate_signal(req: ValidateSignalRequest):
    """
    Validate a trading signal against the current regime strategy profile.
    Returns: approved, adjusted_size, tier_override, signal_state, entry_notes.
    """
    from strategy_engine.regime_adaptive import RegimeAdaptiveEngine
    engine   = RegimeAdaptiveEngine()
    decision = engine.validate_signal(
        symbol             = req.symbol.upper(),
        market_state       = req.market_state.upper(),
        confluence_score   = req.confluence_score,
        setup_type         = req.setup_type,
        base_position_size = req.base_position_size,
        account_equity     = req.account_equity,
    )
    return decision


@router.get("/opportunity-map")
def get_opportunity_map():
    """
    Group all watchlist instruments by their current market regime.
    Returns: {EXPANSION: [...], TREND: [...], COMPRESSION: [...], CHAOS: [...]}
    """
    from strategy_engine.regime_adaptive import RegimeAdaptiveEngine
    engine = RegimeAdaptiveEngine()
    omap   = engine.get_opportunity_map()
    return {
        "opportunity_map": omap,
        "counts": {state: len(syms) for state, syms in omap.items()},
    }


@router.get("/regimes")
def get_all_profiles():
    """Return all four regime strategy profiles with their parameters."""
    from strategy_engine.regime_adaptive import REGIME_PROFILES
    return {
        "profiles": {
            state: _profile_to_dict(profile)
            for state, profile in REGIME_PROFILES.items()
        }
    }


# ── Internal ───────────────────────────────────────────────────────────────────

def _profile_to_dict(profile) -> dict:
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
