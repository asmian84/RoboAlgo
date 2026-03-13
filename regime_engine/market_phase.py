"""
RoboAlgo - Market Phase Engine
Classifies the current market into one of 6 cycle phases using feature vectors
and cycle metrics.

6-Phase Cycle Model:
  1. Accumulation    — base building after decline; oversold, low volatility
  2. Early Bull      — first impulse higher; momentum turning positive
  3. Momentum Bull   — strong trend, high confidence continuation
  4. Distribution    — weakening internals, divergences forming
  5. Early Bear      — trend broken, selling pressure building
  6. Capitulation    — extreme fear, oversold, reversal setups emerge

Signal thresholds by phase:
  Accumulation   ≥ 70%
  Early Bull     ≥ 80%
  Momentum Bull  ≥ 90%   (strict — only high-probability continuation)
  Distribution   ≥ 80%
  Early Bear     ≥ 90%
  Capitulation   ≥ 70%
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Phase names (canonical strings stored in DB)
ACCUMULATION   = "Accumulation"
EARLY_BULL     = "Early Bull"
MOMENTUM_BULL  = "Momentum Bull"
DISTRIBUTION   = "Distribution"
EARLY_BEAR     = "Early Bear"
CAPITULATION   = "Capitulation"

ALL_PHASES = [ACCUMULATION, EARLY_BULL, MOMENTUM_BULL, DISTRIBUTION, EARLY_BEAR, CAPITULATION]

# Minimum probability threshold per phase
PHASE_THRESHOLDS = {
    ACCUMULATION:  0.70,
    EARLY_BULL:    0.80,
    MOMENTUM_BULL: 0.90,
    DISTRIBUTION:  0.80,
    EARLY_BEAR:    0.90,
    CAPITULATION:  0.70,
}

# Confidence tier cutoffs (applied after phase threshold check)
TIER_HIGH   = 0.90
TIER_MEDIUM = 0.70
TIER_LOW    = 0.50


def classify_phase(features: dict) -> tuple[str, float]:
    """
    Classify the market phase for a single instrument on a single date.

    Args:
        features: dict with keys matching Feature model columns:
            trend_strength, momentum, volatility_percentile, volume_ratio,
            cycle_phase, macd_norm, bb_position, price_to_ma50,
            return_5d, return_20d

    Returns:
        (phase_name, confidence_score 0-1)
    """
    trend     = features.get("trend_strength") or 0.0
    momentum  = features.get("momentum") or 0.5      # RSI/100, 0-1
    vol_pct   = features.get("volatility_percentile") or 1.0
    macd      = features.get("macd_norm") or 0.0
    bb_pos    = features.get("bb_position") or 0.5
    r5d       = features.get("return_5d") or 0.0
    r20d      = features.get("return_20d") or 0.0
    cyc       = features.get("cycle_phase") or 0.5   # 0-1

    # Derived signals
    rsi_val   = momentum * 100     # back to RSI scale
    oversold  = rsi_val < 35
    overbought = rsi_val > 65
    trend_pos = trend > 0.02
    trend_neg = trend < -0.02
    vol_high  = vol_pct > 1.3
    vol_low   = vol_pct < 0.8
    macd_pos  = macd > 0.01
    macd_neg  = macd < -0.01
    price_expanding = r5d > 0.01 and r20d > 0.02
    price_contracting = r5d < -0.01 and r20d < -0.02

    # ── Phase classification ─────────────────────────────────────────────

    # CAPITULATION: extreme oversold + high vol + negative trend
    if oversold and trend_neg and vol_high and r20d < -0.05:
        score = _score([
            oversold, trend_neg, vol_high, r20d < -0.05, macd_neg
        ])
        return CAPITULATION, score

    # ACCUMULATION: oversold/neutral, vol falling, flat/slightly neg trend
    if rsi_val < 50 and not trend_pos and vol_pct <= 1.2 and not price_expanding:
        score = _score([
            rsi_val < 50, not trend_pos, vol_pct <= 1.2,
            bb_pos < 0.5, not price_expanding
        ])
        return ACCUMULATION, score

    # EARLY BEAR: trend just turned negative, momentum falling
    if trend_neg and not oversold and macd_neg:
        score = _score([
            trend_neg, macd_neg, r5d < 0, r20d < 0, not oversold
        ])
        return EARLY_BEAR, score

    # DISTRIBUTION: trend still positive but breadth deteriorating
    if trend_pos and (overbought or vol_high or (macd_neg and r5d < 0)):
        score = _score([
            trend_pos, overbought or vol_high,
            macd_neg or r5d < 0, bb_pos > 0.6
        ])
        return DISTRIBUTION, score

    # MOMENTUM BULL: strong trend, momentum strong, low vol
    if trend_pos and rsi_val >= 55 and vol_low and macd_pos and price_expanding:
        score = _score([
            trend > 0.05, rsi_val >= 55, vol_low, macd_pos, price_expanding
        ])
        return MOMENTUM_BULL, score

    # EARLY BULL: positive momentum turning, trend just positive
    if (trend_pos or macd_pos) and rsi_val >= 45:
        score = _score([
            trend_pos or macd_pos, rsi_val >= 45,
            r5d > 0, bb_pos >= 0.4
        ])
        return EARLY_BULL, score

    # Default fallback: Accumulation
    return ACCUMULATION, 0.60


def _score(conditions: list[bool]) -> float:
    """Convert a list of boolean conditions into a confidence score 0.6-0.99."""
    n = len(conditions)
    met = sum(1 for c in conditions if c)
    # Scale from 0.60 (1 met) to 0.99 (all met)
    raw = met / n
    return round(0.60 + raw * 0.39, 3)


def classify_phase_series(feature_df: pd.DataFrame) -> pd.Series:
    """
    Classify phase for every row in a feature DataFrame.

    Args:
        feature_df: DataFrame with feature columns, date index.

    Returns:
        Series of phase names, indexed by date.
    """
    phases = {}
    for dt, row in feature_df.iterrows():
        phase, _ = classify_phase(row.to_dict())
        phases[dt] = phase
    return pd.Series(phases, name="market_phase")


def get_threshold(phase: str) -> float:
    """Return the minimum probability threshold for a given phase."""
    return PHASE_THRESHOLDS.get(phase, 0.80)


def get_confidence_tier(probability: float) -> str | None:
    """
    Classify a probability into HIGH / MEDIUM / LOW.
    Returns None if below the minimum (< 50%).
    """
    if probability >= TIER_HIGH:
        return "HIGH"
    elif probability >= TIER_MEDIUM:
        return "MEDIUM"
    elif probability >= TIER_LOW:
        return "LOW"
    return None


def signal_qualifies(probability: float, phase: str) -> bool:
    """
    Check if a probability meets the threshold for the given market phase.
    The signal must also be above the 50% discard floor.
    """
    tier = get_confidence_tier(probability)
    if tier is None:
        return False
    return probability >= get_threshold(phase)
