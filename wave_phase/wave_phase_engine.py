"""Wave Phase Filter — classifies price action as IMPULSE or CORRECTION.

Definitions
-----------
IMPULSE phase
    Strong directional movement: ATR is expanding AND price structure is
    trending (Higher Highs / Higher Lows or Lower Highs / Lower Lows).
    Reversal trades carry significantly higher risk during an impulse.

CORRECTION phase
    Pullback or consolidation: ATR is contracting or normal AND swing range
    is compressing.  This is the optimal window for reversal setups.

TRANSITION (uncertain)
    Signals are mixed — e.g. expanding ATR but no structural trend, or
    trend structure present but ATR contracting.  Lower confidence score.

Detection logic
---------------
Two independent signal axes are combined:

1. **ATR state** (consumed from market_regime.volatility_state via the
   regime dict, or derived from the vol engine).  No ATR recomputation.
   EXPANDING  → impulse evidence  (+)
   CONTRACTING → correction evidence (+)
   NORMAL      → neutral

2. **Swing amplitude** (computed locally from swing pivot prices supplied
   by structure_engine.swing_detector).  No raw indicator recomputation.
   The ratio of the last swing range to the previous swing range determines
   whether amplitude is expanding (impulse) or contracting (correction).

Confidence
----------
confidence = weighted blend of:
    40% ATR state strength   (how extreme is the ATR ratio)
    40% swing amplitude ratio (how much compression/expansion vs prior)
    20% trend alignment      (does the regime trend match the expected phase)

IMPORTANT: This module does NOT call ATR, EMA, or any indicator.  It
consumes regime dicts and swing pivot lists produced by upstream engines.

Output
------
{
    "wave_phase":    "IMPULSE" | "CORRECTION" | "TRANSITION",
    "direction":     "UP" | "DOWN" | "NEUTRAL",  # of the detected phase
    "confidence":    float,                       # 0–1
    "atr_state":     str,                         # "EXPANDING" | "NORMAL" | "CONTRACTING"
    "amplitude_ratio": float,                     # last_range / prev_range
    "swing_count":   int,                         # pivots available for analysis
}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Amplitude ratio thresholds ─────────────────────────────────────────────────
_EXPANSION_RATIO    = 1.20   # last swing range ≥ 1.2× prior → expanding
_CONTRACTION_RATIO  = 0.80   # last swing range ≤ 0.8× prior → compressing
_MIN_PIVOTS         = 4      # need at least 4 swing points (2 highs + 2 lows)


class WavePhaseEngine:
    """Classify the current market wave phase (IMPULSE vs CORRECTION).

    Parameters
    ----------
    symbol:
        Ticker symbol for logging context.
    min_pivots:
        Minimum swing pivots required to compute amplitude ratio.
    """

    def __init__(self, symbol: str = "", min_pivots: int = _MIN_PIVOTS) -> None:
        self.symbol     = symbol
        self.min_pivots = min_pivots

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Classify the wave phase from OHLCV data.

        Consumes:
            • structure_engine.swing_detector  → swing pivot lists
            • market_regime.MarketRegimeEngine → regime + vol state

        Args:
            df: OHLCV DataFrame with at minimum ``close``, ``high``, ``low``.

        Returns:
            Wave phase dict (see module docstring for schema).
        """
        if len(df) < 20:
            return self._default()

        # ── Consume engine outputs ────────────────────────────────────────────
        regime = self._get_regime(df)
        swings = self._get_swings(df)

        swing_highs = swings.get("swing_highs", [])
        swing_lows  = swings.get("swing_lows",  [])

        atr_state   = regime.get("volatility_state", "NORMAL")
        trend       = regime.get("trend",            "NEUTRAL")
        trend_str   = regime.get("trend_strength",   0.0)
        atr_ratio   = regime.get("atr_ratio",        1.0)

        # ── Swing amplitude ratio ─────────────────────────────────────────────
        amplitude_ratio, swing_count = _compute_amplitude_ratio(
            swing_highs, swing_lows
        )

        # ── Classify phase ────────────────────────────────────────────────────
        phase, direction, confidence = _classify_phase(
            atr_state=atr_state,
            atr_ratio=atr_ratio,
            amplitude_ratio=amplitude_ratio,
            trend=trend,
            trend_strength=trend_str,
            swing_count=swing_count,
            min_pivots=self.min_pivots,
        )

        result = {
            "wave_phase":      phase,
            "direction":       direction,
            "confidence":      round(confidence, 4),
            "atr_state":       atr_state,
            "amplitude_ratio": round(amplitude_ratio, 4),
            "swing_count":     swing_count,
        }

        logger.info(
            "WavePhaseEngine[%s]: phase=%s dir=%s conf=%.2f atr=%s amp_ratio=%.2f",
            self.symbol, phase, direction, confidence, atr_state, amplitude_ratio,
        )
        return result

    # ── Engine Adapters ────────────────────────────────────────────────────────

    def _get_regime(self, df: pd.DataFrame) -> dict:
        """Consume regime + vol state from MarketRegimeEngine."""
        try:
            from market_regime import MarketRegimeEngine
            return MarketRegimeEngine(symbol=self.symbol).run(df)
        except Exception as exc:
            logger.warning("WavePhaseEngine[%s]: market_regime unavailable: %s",
                           self.symbol, exc)
            return {
                "volatility_state": "NORMAL",
                "trend": "NEUTRAL",
                "trend_strength": 0.0,
                "atr_ratio": 1.0,
            }

    def _get_swings(self, df: pd.DataFrame) -> dict:
        """Consume swing pivots from structure_engine.swing_detector."""
        try:
            from structure_engine.swing_detector import (
                compute_adaptive_minimum_move,
                detect_swings,
            )
            mm = compute_adaptive_minimum_move(df)
            return detect_swings(df, minimum_move=mm)
        except Exception as exc:
            logger.warning("WavePhaseEngine[%s]: swing_detector unavailable: %s",
                           self.symbol, exc)
            return {"swing_highs": [], "swing_lows": []}

    def _default(self) -> dict[str, Any]:
        return {
            "wave_phase":      "TRANSITION",
            "direction":       "NEUTRAL",
            "confidence":      0.0,
            "atr_state":       "NORMAL",
            "amplitude_ratio": 1.0,
            "swing_count":     0,
        }


# ── Pure classification functions (no engine calls) ────────────────────────────

def _compute_amplitude_ratio(
    swing_highs: list[dict],
    swing_lows: list[dict],
) -> tuple[float, int]:
    """Compute the ratio of the last swing range to the prior swing range.

    Swing range = distance between consecutive swing high and swing low.
    A ratio > 1 means the most recent swing was larger (expansion).
    A ratio < 1 means the most recent swing was smaller (compression).

    Returns:
        (amplitude_ratio, pivot_count)
        ratio = 1.0 when insufficient pivots are available.
    """
    # Merge both pivot types into a single timeline
    pivots: list[tuple[int, float, str]] = []
    for sh in swing_highs:
        idx = sh.get("index", -1)
        if idx >= 0:
            pivots.append((idx, float(sh["price"]), "high"))
    for sl in swing_lows:
        idx = sl.get("index", -1)
        if idx >= 0:
            pivots.append((idx, float(sl["price"]), "low"))

    pivots.sort(key=lambda p: p[0])
    n = len(pivots)

    if n < 4:
        return 1.0, n

    # Compute alternating HH-LL swing ranges from the right (most recent first)
    ranges: list[float] = []
    for i in range(n - 1, 0, -1):
        r = abs(pivots[i][1] - pivots[i - 1][1])
        if r > 0:
            ranges.append(r)
        if len(ranges) == 2:
            break

    if len(ranges) < 2:
        return 1.0, n

    # last range / prior range
    amplitude_ratio = ranges[0] / max(ranges[1], 1e-9)
    return amplitude_ratio, n


def _classify_phase(
    atr_state: str,
    atr_ratio: float,
    amplitude_ratio: float,
    trend: str,
    trend_strength: float,
    swing_count: int,
    min_pivots: int,
) -> tuple[str, str, float]:
    """Apply the decision rules and return (phase, direction, confidence).

    Priority
    --------
    1. IMPULSE:    ATR EXPANDING  AND amplitude expanding   AND trend present
    2. CORRECTION: ATR CONTRACTING OR  amplitude compressing AND trend weak/neutral
    3. TRANSITION: mixed / insufficient pivots
    """
    direction = trend if trend != "NEUTRAL" else "NEUTRAL"

    # ── IMPULSE conditions ────────────────────────────────────────────────────
    atr_expanding   = atr_state == "EXPANDING"
    atr_contracting = atr_state == "CONTRACTING"
    amp_expanding   = amplitude_ratio >= _EXPANSION_RATIO
    amp_contracting = amplitude_ratio <= _CONTRACTION_RATIO
    trend_present   = trend != "NEUTRAL" and trend_strength >= 0.3
    enough_pivots   = swing_count >= min_pivots

    if atr_expanding and amp_expanding and trend_present:
        # Strong impulse: both axes confirming expansion + structure
        atr_score = min((atr_ratio - 1.5) / 1.5, 1.0)           # 0→1 as ratio 1.5→3.0
        amp_score = min((amplitude_ratio - 1.0) / 1.0, 1.0)     # 0→1 as ratio 1→2
        trd_score = min(trend_strength, 1.0)
        confidence = 0.40 * atr_score + 0.40 * amp_score + 0.20 * trd_score
        return "IMPULSE", direction, max(confidence, 0.5)

    if atr_expanding and trend_present:
        # ATR expansion with structure but no amplitude confirmation yet
        atr_score  = min((atr_ratio - 1.5) / 1.5, 1.0)
        confidence = 0.50 * atr_score + 0.30 * trend_strength + 0.20 * 0.5
        return "IMPULSE", direction, max(confidence, 0.35)

    # ── CORRECTION conditions ─────────────────────────────────────────────────
    if atr_contracting and amp_contracting:
        # Both axes contracting → high-confidence correction
        atr_score  = min((1.0 - atr_ratio) / 0.3, 1.0)          # 0→1 as ratio 0.7→0.4
        amp_score  = min((1.0 - amplitude_ratio) / 0.2, 1.0)    # 0→1 as ratio 0.8→0.6
        trd_score  = 1.0 - trend_strength                        # weaker trend = more correction
        confidence = 0.40 * atr_score + 0.40 * amp_score + 0.20 * trd_score
        return "CORRECTION", direction, max(confidence, 0.5)

    if amp_contracting and not atr_expanding:
        # Swing compression without ATR expansion
        amp_score  = min((1.0 - amplitude_ratio) / 0.2, 1.0)
        confidence = 0.60 * amp_score + 0.40 * (1.0 - trend_strength)
        return "CORRECTION", direction, max(confidence, 0.35)

    if atr_contracting and not amp_expanding:
        # ATR contraction without amplitude expansion
        atr_score  = min((1.0 - atr_ratio) / 0.3, 1.0)
        confidence = 0.60 * atr_score + 0.40 * (1.0 - trend_strength)
        return "CORRECTION", direction, max(confidence, 0.3)

    # ── TRANSITION: mixed signals ─────────────────────────────────────────────
    return "TRANSITION", direction, 0.25 if enough_pivots else 0.10


# ── Convenience function ───────────────────────────────────────────────────────

def detect_wave_phase(
    df: pd.DataFrame,
    symbol: str = "",
) -> dict[str, Any]:
    """Functional wrapper around :class:`WavePhaseEngine`.

    Args:
        df:     OHLCV DataFrame.
        symbol: Optional symbol for logging.

    Returns:
        Wave phase dict.
    """
    return WavePhaseEngine(symbol=symbol).run(df)
