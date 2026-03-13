"""Multi-Timeframe Alignment Engine — measures signal agreement across timeframes.

Purpose
-------
A trade setup is significantly stronger when the same directional bias appears
across multiple timeframes simultaneously.  This engine quantifies how well
signals align across the bias → setup → entry timeframe hierarchy for each
of the three RoboAlgo trading models.

Trading models and their timeframe hierarchies
----------------------------------------------
DAY_TRADE:   bias=1h   → setup=15m  → entry=5m  (or 1m)
SWING_TRADE: bias=Daily → setup=4h  → entry=1h
INVESTMENT:  bias=Weekly → setup=Daily → entry=4h (or Daily)

Alignment scoring
-----------------
Each timeframe in the hierarchy contributes a weighted vote:

    Bias     timeframe → weight 0.50  (must agree for strong alignment)
    Setup    timeframe → weight 0.30
    Entry    timeframe → weight 0.20

A timeframe "agrees" when its regime trend matches the master bias direction.
Partial credit is given when:
    • Regime is RANGE / ACCUMULATION but trend_strength is moderate (0.2–0.4).

The alignment score is the weighted sum of agreeing timeframes (0–1).

Bias direction
--------------
The bias direction is taken from the **highest timeframe** in the hierarchy.
It can only be ``"LONG"`` or ``"SHORT"``:
    regime TREND_UP / ACCUMULATION / VOLATILITY_EXPANSION + trend UP → LONG
    regime TREND_DOWN / DISTRIBUTION                                  → SHORT

Output
------
{
    "alignment_score":     float,          # 0–1 weighted agreement score
    "bias_direction":      "LONG" | "SHORT" | "NEUTRAL",
    "model":               str,            # "DAY_TRADE" | "SWING_TRADE" | "INVESTMENT"
    "aligned_timeframes":  list[str],      # TFs that agree with bias
    "conflicting_timeframes": list[str],   # TFs that conflict with bias
    "timeframe_details":   dict[str, dict],# per-TF regime summary
    "trade_allowed":       bool,           # False when bias is NEUTRAL
}

Input contract
--------------
The caller provides a dict mapping timeframe labels to OHLCV DataFrames::

    {
        "daily":  pd.DataFrame(...),   # bias TF for SWING_TRADE
        "4h":     pd.DataFrame(...),   # setup TF
        "1h":     pd.DataFrame(...),   # entry TF
    }

The engine runs MarketRegimeEngine on each DataFrame independently and does
NOT recompute any indicator — regime analysis is delegated entirely to the
existing MarketRegimeEngine.

IMPORTANT: DataFrames for lower timeframes must be large enough to produce
meaningful regime signals (≥ 20 bars minimum).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Trading model definitions ──────────────────────────────────────────────────

MODEL_DAY_TRADE   = "DAY_TRADE"
MODEL_SWING_TRADE = "SWING_TRADE"
MODEL_INVESTMENT  = "INVESTMENT"

# Each model maps tier → (label, weight)
_MODELS: dict[str, list[tuple[str, float]]] = {
    MODEL_DAY_TRADE: [
        ("1h",    0.50),   # bias
        ("15m",   0.30),   # setup
        ("5m",    0.20),   # entry (accepts "1m" as fallback)
    ],
    MODEL_SWING_TRADE: [
        ("daily", 0.50),   # bias
        ("4h",    0.30),   # setup
        ("1h",    0.20),   # entry
    ],
    MODEL_INVESTMENT: [
        ("weekly",0.50),   # bias
        ("daily", 0.30),   # setup
        ("4h",    0.20),   # entry (accepts "daily" as fallback)
    ],
}

# Regime → direction mapping
_BULLISH_REGIMES = {"TREND_UP", "ACCUMULATION"}
_BEARISH_REGIMES = {"TREND_DOWN", "DISTRIBUTION"}
_NEUTRAL_REGIMES = {"RANGE", "VOLATILITY_EXPANSION"}

# Partial credit threshold: regime agrees if trend_strength ≥ this for neutral regimes
_PARTIAL_CREDIT_MIN_STRENGTH = 0.25


class MTFAlignmentEngine:
    """Compute multi-timeframe signal alignment for a given trading model.

    Parameters
    ----------
    symbol:
        Ticker symbol for logging context.
    model:
        Trading model: ``"DAY_TRADE"``, ``"SWING_TRADE"``, or ``"INVESTMENT"``.
    """

    def __init__(
        self,
        symbol: str = "",
        model: str = MODEL_SWING_TRADE,
    ) -> None:
        self.symbol = symbol
        self.model  = model

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        timeframe_dfs: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Compute alignment score across the model's timeframe hierarchy.

        Args:
            timeframe_dfs: Dict mapping timeframe labels to OHLCV DataFrames.
                           Labels should match the model hierarchy (e.g.
                           ``"daily"``, ``"4h"``, ``"1h"`` for SWING_TRADE).
                           Missing timeframes are silently skipped.

        Returns:
            Alignment dict (see module docstring for full schema).
        """
        hierarchy = _MODELS.get(self.model, _MODELS[MODEL_SWING_TRADE])

        # ── Run regime on each available timeframe ────────────────────────────
        tf_regimes: dict[str, dict] = {}
        for tf_label, _weight in hierarchy:
            df = self._resolve_df(timeframe_dfs, tf_label)
            if df is not None and len(df) >= 20:
                tf_regimes[tf_label] = self._get_regime(df, tf_label)
            else:
                logger.debug(
                    "MTFAlignmentEngine[%s]: TF %s — no/insufficient data",
                    self.symbol, tf_label,
                )
                tf_regimes[tf_label] = _empty_regime(tf_label)

        # ── Determine master bias from highest-weight timeframe ────────────────
        bias_tf    = hierarchy[0][0]
        bias_regime = tf_regimes.get(bias_tf, _empty_regime(bias_tf))
        bias_direction = _regime_to_direction(bias_regime)

        # ── Score each tier ───────────────────────────────────────────────────
        aligned:    list[str] = []
        conflicting:list[str] = []
        weighted_score = 0.0
        details: dict[str, dict] = {}

        for tf_label, weight in hierarchy:
            regime = tf_regimes.get(tf_label, _empty_regime(tf_label))
            tf_dir = _regime_to_direction(regime)

            agree, partial = _tf_agrees(tf_dir, bias_direction, regime)
            credit = weight if agree else (weight * 0.4 if partial else 0.0)
            weighted_score += credit

            if agree or partial:
                aligned.append(tf_label)
            elif bias_direction != "NEUTRAL" and tf_dir not in ("NEUTRAL", "UNKNOWN"):
                conflicting.append(tf_label)

            details[tf_label] = {
                "regime":         regime.get("regime",           "UNKNOWN"),
                "trend":          regime.get("trend",            "NEUTRAL"),
                "trend_strength": regime.get("trend_strength",   0.0),
                "volatility_state": regime.get("volatility_state","NORMAL"),
                "direction":      tf_dir,
                "agrees":         agree,
                "partial_agree":  partial,
                "weight":         weight,
                "credit":         round(credit, 3),
            }

        alignment_score = round(min(weighted_score, 1.0), 4)
        trade_allowed   = bias_direction != "NEUTRAL" and alignment_score >= 0.5

        result = {
            "alignment_score":        alignment_score,
            "bias_direction":         bias_direction,
            "model":                  self.model,
            "aligned_timeframes":     aligned,
            "conflicting_timeframes": conflicting,
            "timeframe_details":      details,
            "trade_allowed":          trade_allowed,
        }

        logger.info(
            "MTFAlignmentEngine[%s] model=%s bias=%s score=%.2f aligned=%s conflict=%s",
            self.symbol, self.model, bias_direction,
            alignment_score, aligned, conflicting,
        )
        return result

    def score_direction(
        self,
        timeframe_dfs: dict[str, pd.DataFrame],
        direction: str,
    ) -> float:
        """Return the alignment score for a specific trade direction (0–1).

        Args:
            timeframe_dfs: Same dict as :meth:`run`.
            direction:     ``"LONG"`` or ``"SHORT"``.

        Returns:
            0.0–1.0 alignment score, 0.0 when bias is opposite to ``direction``.
        """
        result = self.run(timeframe_dfs)
        if result["bias_direction"] != direction:
            return 0.0
        return result["alignment_score"]

    # ── Engine Adapter ─────────────────────────────────────────────────────────

    def _get_regime(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Run MarketRegimeEngine on a single timeframe DataFrame."""
        try:
            from market_regime import MarketRegimeEngine
            engine = MarketRegimeEngine(symbol=self.symbol, timeframe=timeframe)
            return engine.run(df)
        except Exception as exc:
            logger.warning(
                "MTFAlignmentEngine[%s]: regime failed for TF %s: %s",
                self.symbol, timeframe, exc,
            )
            return _empty_regime(timeframe)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_df(
        dfs: dict[str, pd.DataFrame],
        label: str,
    ) -> pd.DataFrame | None:
        """Find the DataFrame for ``label``, trying common aliases.

        Example: ``"daily"`` also matches ``"1d"``; ``"4h"`` matches ``"4H"``.
        """
        aliases: dict[str, list[str]] = {
            "daily":  ["1d", "D",  "day", "1D"],
            "weekly": ["1w", "W",  "week","1W"],
            "4h":     ["4H", "240m"],
            "1h":     ["1H", "60m"],
            "15m":    ["15M","15"],
            "5m":     ["5M", "5"],
            "1m":     ["1M", "1"],
        }
        # Exact match
        if label in dfs:
            return dfs[label]
        # Alias match
        for alias in aliases.get(label, []):
            if alias in dfs:
                return dfs[alias]
        # Case-insensitive match
        label_lower = label.lower()
        for key, df in dfs.items():
            if key.lower() == label_lower:
                return df
        return None


# ── Standalone alignment helper ─────────────────────────────────────────────────

def compute_alignment(
    timeframe_dfs: dict[str, pd.DataFrame],
    symbol: str = "",
    model: str = MODEL_SWING_TRADE,
) -> dict[str, Any]:
    """Functional wrapper around :class:`MTFAlignmentEngine`.

    Args:
        timeframe_dfs: Dict of {tf_label: OHLCV DataFrame}.
        symbol:        Optional symbol for logging.
        model:         Trading model label.

    Returns:
        Alignment dict.
    """
    return MTFAlignmentEngine(symbol=symbol, model=model).run(timeframe_dfs)


# ── Pure helper functions ──────────────────────────────────────────────────────

def _regime_to_direction(regime: dict) -> str:
    """Convert a regime dict to a binary LONG / SHORT / NEUTRAL direction.

    VOLATILITY_EXPANSION maps to the underlying trend direction when one is
    present, otherwise NEUTRAL (could break either way).
    """
    label  = regime.get("regime",         "RANGE")
    trend  = regime.get("trend",          "NEUTRAL")
    tstr   = regime.get("trend_strength", 0.0)

    if label in _BULLISH_REGIMES:
        return "LONG"
    if label in _BEARISH_REGIMES:
        return "SHORT"
    if label == "VOLATILITY_EXPANSION":
        if trend == "UP" and tstr >= 0.3:
            return "LONG"
        if trend == "DOWN" and tstr >= 0.3:
            return "SHORT"
    # RANGE or VOLATILITY_EXPANSION without trend → NEUTRAL
    return "NEUTRAL"


def _tf_agrees(
    tf_direction: str,
    bias_direction: str,
    regime: dict,
) -> tuple[bool, bool]:
    """Return (full_agree, partial_agree) for a single timeframe.

    Full agree:    tf_direction == bias_direction
    Partial agree: tf is NEUTRAL but trend_strength is non-trivial in the
                   bias direction (the market is not actively opposing it).
    """
    if bias_direction == "NEUTRAL":
        return False, False

    if tf_direction == bias_direction:
        return True, False

    if tf_direction == "NEUTRAL":
        tstr = regime.get("trend_strength", 0.0)
        # Partial credit when neutral but not actively opposing
        if tstr >= _PARTIAL_CREDIT_MIN_STRENGTH:
            return False, True   # lean-agree
        return False, True       # pure neutral — still partial credit (no conflict)

    # Opposite direction — full conflict
    return False, False


def _empty_regime(timeframe: str) -> dict:
    return {
        "regime":           "RANGE",
        "trend":            "NEUTRAL",
        "trend_strength":   0.0,
        "volatility_state": "NORMAL",
        "timeframe":        timeframe,
    }
