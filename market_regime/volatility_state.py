"""Volatility State — classifies the current volatility regime.

This module consumes ATR values from the volatility_engine rather than
recomputing them.  It only classifies the *state* of volatility based on
comparing current ATR to its rolling mean.

States
------
EXPANDING  — current ATR is significantly above its historical mean
NORMAL     — current ATR is within a normal band around its mean
CONTRACTING — current ATR is significantly below its historical mean

The classification directly gates the VOLATILITY_EXPANSION regime
in the Market Regime Engine and contributes to ACCUMULATION/DISTRIBUTION
detection.

IMPORTANT: ATR values are consumed from volatility_engine via the caller
(regime_engine.py).  This module performs NO indicator recalculation.
"""

from __future__ import annotations


# Multiplier thresholds for ATR state classification
_EXPANSION_MULTIPLIER    = 1.5   # ATR > 1.5× mean → EXPANDING
_CONTRACTION_MULTIPLIER  = 0.7   # ATR < 0.7× mean → CONTRACTING


def classify_volatility_state(
    current_atr: float,
    atr_history: list[float],
    expansion_mult: float = _EXPANSION_MULTIPLIER,
    contraction_mult: float = _CONTRACTION_MULTIPLIER,
) -> dict:
    """Classify the current ATR level relative to its historical average.

    Args:
        current_atr:      The ATR value at the current bar.
        atr_history:      List of recent ATR values (oldest first).
                          Used to compute the reference mean.
        expansion_mult:   ATR/mean ratio above which state is EXPANDING.
        contraction_mult: ATR/mean ratio below which state is CONTRACTING.

    Returns:
        dict::

            {
                "volatility_state": "EXPANDING" | "NORMAL" | "CONTRACTING",
                "atr_ratio":        float,   # current_atr / mean_atr
                "current_atr":      float,
                "mean_atr":         float,
                "is_expansion":     bool,
                "is_contraction":   bool,
            }
    """
    if not atr_history or current_atr <= 0:
        return _unknown_state(current_atr)

    valid_history = [v for v in atr_history if v > 0]
    if not valid_history:
        return _unknown_state(current_atr)

    mean_atr = sum(valid_history) / len(valid_history)
    if mean_atr == 0:
        return _unknown_state(current_atr)

    atr_ratio = current_atr / mean_atr

    if atr_ratio >= expansion_mult:
        state = "EXPANDING"
    elif atr_ratio <= contraction_mult:
        state = "CONTRACTING"
    else:
        state = "NORMAL"

    return {
        "volatility_state": state,
        "atr_ratio":        round(atr_ratio, 4),
        "current_atr":      round(current_atr, 6),
        "mean_atr":         round(mean_atr, 6),
        "is_expansion":     state == "EXPANDING",
        "is_contraction":   state == "CONTRACTING",
    }


def describe_volatility(vol_dict: dict) -> str:
    """Return a human-readable one-line description of the volatility state."""
    state = vol_dict.get("volatility_state", "UNKNOWN")
    ratio = vol_dict.get("atr_ratio", 0.0)
    return f"{state} (ATR ratio: {ratio:.2f}×)"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _unknown_state(current_atr: float) -> dict:
    return {
        "volatility_state": "NORMAL",
        "atr_ratio":        1.0,
        "current_atr":      round(current_atr, 6),
        "mean_atr":         round(current_atr, 6),
        "is_expansion":     False,
        "is_contraction":   False,
    }
