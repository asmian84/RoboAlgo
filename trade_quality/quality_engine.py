"""Trade Quality Scoring Engine — main orchestrator.

Evaluates every trade setup produced by TradeSetupEngine and assigns a
quality grade (A+, A, B, C) based on multi-engine confluence scoring.

Consuming engines (signals consumed, NOT recomputed)
----------------------------------------------------
    • reversal_sniper.ReversalSniperEngine    — sniper signal presence
    • liquidity_map.LiquidityMapEngine        — swept zones, zone proximity
    • market_regime.MarketRegimeEngine        — regime alignment
    • cycle_engine (optional)                 — cycle peak/trough exhaustion
    • pattern_engine.chart_patterns (optional)— corrective wave confirmation

Output format per setup
-----------------------
{
    "setup_quality":       "A+",    # letter grade
    "score":               11,      # raw points
    "max_score":           13,
    "trade_allowed":       True,
    "confidence":          0.85,    # score / max_score
    "position_multiplier": 1.0,     # grade-adjusted size factor
    "factors_met":         ["SNIPER_REVERSAL", "LIQUIDITY_SWEEP", ...],
    "factors_missed":      ["TIME_EXHAUSTION"],
    # Original setup fields included for convenience:
    "trade_type":          "SWING_TRADE",
    "setup":               "LIQUIDITY_REVERSAL",
    "direction":           "LONG",
    "entry":               21.15,
    "stop_loss":           20.95,
    "targets":             [21.80, 22.10, 22.60],
    "risk_reward":         3.2,
    "position_size_adj":   float,   # position_sizer output × multiplier
    "symbol":              "TQQQ",
    "date":                "2025-03-09",
}
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from trade_quality.scoring_factors import (
    FACTORS,
    MAX_SCORE,
    evaluate_factors,
)
from trade_quality.grade_classifier import (
    classify_grade,
    compute_confidence,
    adjust_position_size,
    grade_summary,
)

logger = logging.getLogger(__name__)


class TradeQualityEngine:
    """Score and grade trade setups from the Trade Setup Engine.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"TQQQ"``).
    min_grade:
        Minimum grade required to allow a trade.  Setups below this grade
        will have ``trade_allowed = False`` regardless of the score.
        Default ``"B"`` — allows A+, A, B grades.
    """

    def __init__(
        self,
        symbol: str,
        min_grade: str = "B",
    ) -> None:
        self.symbol    = symbol
        self.min_grade = min_grade

    # ── Public API ──────────────────────────────────────────────────────────────

    def score_setups(
        self,
        df: pd.DataFrame,
        setups: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Score all trade setups for the supplied DataFrame.

        Args:
            df:     OHLCV DataFrame.
            setups: Pre-computed list of trade plan dicts from TradeSetupEngine.
                    When *None* the engine generates setups internally.

        Returns:
            List of graded setup dicts, sorted by score descending (best first).
        """
        if setups is None:
            setups = self._get_setups(df)

        if not setups:
            logger.info("TradeQualityEngine[%s]: no setups to score", self.symbol)
            return []

        # ── Shared engine outputs (fetch once, reuse across all setups) ───────
        regime         = self._get_regime(df)
        liq_map        = self._get_liquidity_map(df)
        cycle_data     = self._get_cycle_data(df)
        patterns       = self._get_patterns(df)

        graded: list[dict[str, Any]] = []

        for setup in setups:
            result = self._score_one(
                setup, regime,
                liq_map, cycle_data, patterns,
            )
            graded.append(result)
            logger.info(
                "TradeQualityEngine[%s] %s/%s bar=%d → %s",
                self.symbol,
                setup.get("setup",     ""),
                setup.get("direction", ""),
                setup.get("bar_index", 0),
                grade_summary(
                    classify_grade(result["score"]),
                    result["score"],
                ),
            )

        graded.sort(key=lambda r: r["score"], reverse=True)
        return graded

    # ── Scoring core ────────────────────────────────────────────────────────────

    def _score_one(
        self,
        setup: dict,
        regime: dict,
        liq_map: dict,
        cycle_data: dict | None,
        patterns: list[dict],
    ) -> dict[str, Any]:
        """Evaluate all factors for a single setup and build the graded dict."""
        flags = evaluate_factors(
            setup=setup,
            regime=regime,
            liq_map=liq_map,
            cycle_data=cycle_data,
            patterns=patterns,
        )

        score        = sum(FACTORS[f].weight for f, hit in flags.items() if hit)
        grade_spec   = classify_grade(score, MAX_SCORE)
        confidence   = compute_confidence(score, MAX_SCORE)

        factors_met    = [FACTORS[f].label for f, hit in flags.items() if hit]
        factors_missed = [FACTORS[f].label for f, hit in flags.items() if not hit]

        # Apply min_grade filter
        trade_allowed = grade_spec.trade_allowed and self._meets_min_grade(grade_spec.label)

        # Adjust position size by grade multiplier
        base_pos = setup.get("position", {}).get("position_size", 0.0)
        adj_pos  = adjust_position_size(base_pos, grade_spec)

        return {
            # ── Quality fields ──────────────────────────────────────────────
            "setup_quality":       grade_spec.label,
            "score":               score,
            "max_score":           MAX_SCORE,
            "trade_allowed":       trade_allowed,
            "confidence":          confidence,
            "position_multiplier": grade_spec.position_multiplier,
            "factors_met":         factors_met,
            "factors_missed":      factors_missed,
            # ── Original setup fields (pass-through) ────────────────────────
            "symbol":              setup.get("symbol",      self.symbol),
            "trade_type":          setup.get("trade_type",  ""),
            "setup":               setup.get("setup",       ""),
            "direction":           setup.get("direction",   ""),
            "entry":               setup.get("entry",       0.0),
            "stop_loss":           setup.get("stop_loss",   0.0),
            "targets":             setup.get("targets",     []),
            "risk_reward":         setup.get("risk_reward", 0.0),
            "timeframes":          setup.get("timeframes",  {}),
            "position_size_adj":   adj_pos,
            "date":                setup.get("date",        ""),
            "bar_index":           setup.get("bar_index",   0),
        }

    # ── Engine Adapters (consume — do not reimplement) ─────────────────────────

    def _get_setups(self, df: pd.DataFrame) -> list[dict]:
        """Generate setups via TradeSetupEngine."""
        try:
            from trade_setup import TradeSetupEngine, TRADE_TYPE_SWING
            engine = TradeSetupEngine(symbol=self.symbol, trade_type=TRADE_TYPE_SWING)
            return engine.run(df)
        except Exception as exc:
            logger.warning(
                "TradeQualityEngine[%s]: trade_setup unavailable: %s",
                self.symbol, exc,
            )
            return []

    def _get_regime(self, df: pd.DataFrame) -> dict:
        """Consume regime from market_regime.MarketRegimeEngine."""
        try:
            from market_regime import MarketRegimeEngine
            return MarketRegimeEngine(symbol=self.symbol).run(df)
        except Exception as exc:
            logger.warning(
                "TradeQualityEngine[%s]: market_regime unavailable: %s",
                self.symbol, exc,
            )
            return {"regime": "RANGE", "trend": "NEUTRAL"}

    def _get_liquidity_map(self, df: pd.DataFrame) -> dict:
        """Consume zone data from liquidity_map.LiquidityMapEngine."""
        try:
            from liquidity_map import LiquidityMapEngine
            return LiquidityMapEngine(symbol=self.symbol).run(df)
        except Exception as exc:
            logger.warning(
                "TradeQualityEngine[%s]: liquidity_map unavailable: %s",
                self.symbol, exc,
            )
            return {"liquidity_zones": [], "swept_zones": []}

    def _get_cycle_data(self, df: pd.DataFrame) -> dict | None:
        """Consume cycle peak/trough data from cycle_engine if available."""
        try:
            from cycle_engine.cycle_detector import CycleEngine
            return CycleEngine(symbol=self.symbol).run(df)
        except Exception:
            pass
        try:
            from cycle_engine.wavelet_cycles import detect_wavelet_cycles
            import numpy as np
            close = df["close"].values.astype(float)
            cycle_len, strength = detect_wavelet_cycles(close)
            if cycle_len > 0:
                # Approximate peaks/troughs from cycle length
                n = len(close)
                half = int(cycle_len / 2)
                troughs = list(range(0, n, int(cycle_len)))
                peaks   = list(range(half, n, int(cycle_len)))
                return {
                    "cycle_length": cycle_len,
                    "cycle_strength": strength,
                    "cycle_peaks":   peaks,
                    "cycle_troughs": troughs,
                }
        except Exception as exc:
            logger.warning(
                "TradeQualityEngine[%s]: cycle_engine unavailable: %s",
                self.symbol, exc,
            )
        return None

    def _get_patterns(self, df: pd.DataFrame) -> list[dict]:
        """Consume active patterns from pattern_engine.chart_patterns."""
        try:
            from pattern_engine.chart_patterns import detect_all
            return detect_all(self.symbol, df) or []
        except Exception as exc:
            logger.warning(
                "TradeQualityEngine[%s]: pattern_engine unavailable: %s",
                self.symbol, exc,
            )
            return []

    # ── Helpers ────────────────────────────────────────────────────────────────

    _GRADE_ORDER = {"A+": 4, "A": 3, "B": 2, "C": 1}

    def _meets_min_grade(self, grade_label: str) -> bool:
        """True when ``grade_label`` is ≥ ``self.min_grade``."""
        order = self._GRADE_ORDER
        return order.get(grade_label, 0) >= order.get(self.min_grade, 0)
