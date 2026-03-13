"""Scoring Factors — defines individual condition evaluators for trade quality.

Each factor evaluates one aspect of a trade setup and returns a boolean plus
the point value awarded.  The engine collects all factor results and sums
them into a raw score.

Factor catalogue (max = 10)
---------------------------
Factor                          Weight  Signal source
─────────────────────────────── ──────  ──────────────────────────────────────
liquidity_sweep                   +2   liquidity_map.LiquidityMapEngine
time_exhaustion                   +2   cycle_engine (adaptive cycle peak/trough)
volume_spike                      +1   raw OHLCV volume vs rolling average
higher_tf_sr                      +2   nearest liquidity zone strength ≥ 0.7
wave_phase_correction             +1   pattern_engine (bullish/bearish reversal)
regime_favorable                  +2   market_regime.MarketRegimeEngine

IMPORTANT: This module does NOT recompute any indicator.  All signal checks
are simple boolean tests on data provided by upstream engine outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Factor definitions ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Factor:
    name:   str
    weight: int
    label:  str   # human-readable confluence tag


FACTORS: dict[str, Factor] = {
    "liquidity_sweep":       Factor("liquidity_sweep",       2, "LIQUIDITY_SWEEP"),
    "time_exhaustion":       Factor("time_exhaustion",        2, "TIME_EXHAUSTION"),
    "volume_spike":          Factor("volume_spike",           1, "VOLUME_SPIKE"),
    "higher_tf_sr":          Factor("higher_tf_sr",           2, "HTF_SUPPORT_RESISTANCE"),
    "wave_phase_correction": Factor("wave_phase_correction",  1, "WAVE_CORRECTION"),
    "regime_favorable":      Factor("regime_favorable",       2, "FAVORABLE_REGIME"),
}

MAX_SCORE: int = sum(f.weight for f in FACTORS.values())   # 10


# ── Individual factor evaluators ───────────────────────────────────────────────

def evaluate_factors(
    setup: dict[str, Any],
    regime: dict,
    liq_map: dict,
    cycle_data: dict | None = None,
    patterns: list[dict] | None = None,
) -> dict[str, bool]:
    """Evaluate all scoring factors for a given trade setup.

    Args:
        setup:      Trade plan dict from TradeSetupEngine.
        regime:     Market Regime Engine output dict.
        liq_map:    LiquidityMapEngine output dict.
        cycle_data: Optional CycleEngine output dict.
        patterns:   Optional list of active pattern dicts.

    Returns:
        Dict mapping each factor name to a bool (True = factor scored).
    """
    direction    = setup.get("direction", "LONG")
    bar_index    = setup.get("bar_index", 0)
    entry_price  = setup.get("entry",     0.0)

    return {
        "liquidity_sweep":       _has_liquidity_sweep(liq_map, bar_index),
        "time_exhaustion":       _has_time_exhaustion(cycle_data, bar_index),
        "volume_spike":          _has_volume_spike(setup),
        "higher_tf_sr":          _has_higher_tf_sr(liq_map, entry_price),
        "wave_phase_correction": _has_wave_correction(patterns, direction),
        "regime_favorable":      _has_favorable_regime(regime, direction),
    }


# ── Individual check functions ─────────────────────────────────────────────────

def _has_liquidity_sweep(liq_map: dict, bar_index: int) -> bool:
    """True when a liquidity sweep event occurred at or near the setup bar."""
    swept = liq_map.get("swept_zones", [])
    for event in swept:
        if abs(event.get("bar_index", -999) - bar_index) <= 3:
            return True
    return False


def _has_time_exhaustion(cycle_data: dict | None, bar_index: int) -> bool:
    """True when a cycle engine peak/trough is near the setup bar (±3 bars).

    The cycle engine emits cycle length and phase; exhaustion is inferred
    when the bar is near an expected peak (bearish) or trough (bullish).
    """
    if not cycle_data:
        return False

    # cycle_peaks and cycle_troughs are lists of bar indices
    peaks   = cycle_data.get("cycle_peaks",   [])
    troughs = cycle_data.get("cycle_troughs", [])

    for idx in peaks + troughs:
        if abs(idx - bar_index) <= 3:
            return True
    return False


def _has_volume_spike(setup: dict) -> bool:
    """True when the trade setup was tagged with a VOLUME_SPIKE confluence label."""
    confluence = setup.get("confluence", [])
    if isinstance(confluence, list):
        return "VOLUME_SPIKE" in confluence

    # Fallback: check if sniper signal included volume spike
    sniper = setup.get("_sniper_signal")
    if sniper:
        return "VOLUME_SPIKE" in sniper.get("confluence", [])
    return False


def _has_higher_tf_sr(liq_map: dict, entry_price: float) -> bool:
    """True when a strong (≥ 0.7) liquidity zone is within 1% of entry."""
    zones = liq_map.get("liquidity_zones", [])
    for zone in zones:
        if (
            zone.get("strength", 0) >= 0.7
            and abs(zone.get("price", 0) - entry_price) / max(entry_price, 1e-9) <= 0.01
        ):
            return True
    return False


def _has_wave_correction(patterns: list[dict] | None, direction: str) -> bool:
    """True when a corrective (reversal) chart pattern is active in the trade direction."""
    if not patterns:
        return False

    active_statuses = {"READY", "BREAKOUT", "FORMING"}
    if direction == "LONG":
        wanted = {"bullish"}
    else:
        wanted = {"bearish"}

    for p in patterns:
        if (
            p.get("direction") in wanted
            and p.get("status")    in active_statuses
        ):
            return True
    return False


def _has_favorable_regime(regime: dict, direction: str) -> bool:
    """True when the market regime aligns with the trade direction.

    Favorable regimes:
        LONG  — TREND_UP, ACCUMULATION, VOLATILITY_EXPANSION (potential long breakout)
        SHORT — TREND_DOWN, DISTRIBUTION, VOLATILITY_EXPANSION (potential short breakout)
        RANGE — never favorable (no directional bias)
    """
    regime_label = regime.get("regime", "RANGE")
    trend        = regime.get("trend",  "NEUTRAL")

    if direction == "LONG":
        return regime_label in {"TREND_UP", "ACCUMULATION"} or (
            regime_label == "VOLATILITY_EXPANSION" and trend != "DOWN"
        )
    if direction == "SHORT":
        return regime_label in {"TREND_DOWN", "DISTRIBUTION"} or (
            regime_label == "VOLATILITY_EXPANSION" and trend != "UP"
        )
    return False
