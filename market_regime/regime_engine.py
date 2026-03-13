"""Market Regime Engine — main orchestrator.

Classifies the current market regime by aggregating signals from:
    • structure_engine.swing_detector  — HH/HL/LH/LL swing structure
    • volatility_engine.regime         — ATR values and expansion state
    • liquidity_map                    — equal level sweeps (accum/distrib bias)

Regime labels
-------------
    TREND_UP             Confirmed bullish structure: HH + HL sequence.
    TREND_DOWN           Confirmed bearish structure: LH + LL sequence.
    RANGE                Sideways compression with no directional bias.
    VOLATILITY_EXPANSION ATR spike above norm — breakout imminent.
    ACCUMULATION         Range with bullish liquidity sweep pattern.
    DISTRIBUTION         Range with bearish liquidity sweep pattern.

Output format
-------------
{
    "symbol":           str,
    "regime":           str,      # one of the 6 labels above
    "confidence":       float,    # 0–1 normalised
    "trend":            str,      # "UP" | "DOWN" | "NEUTRAL"
    "trend_strength":   float,    # 0–1
    "volatility_state": str,      # "EXPANDING" | "NORMAL" | "CONTRACTING"
    "is_range":         bool,
    "range_high":       float,
    "range_low":        float,
    "atr_ratio":        float,    # current_atr / mean_atr
    "timeframe":        str,
}

IMPORTANT: No indicators are recomputed here.  ATR, swing detection, and
liquidity sweeps are consumed from existing engines.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from market_regime.trend_detector    import detect_trend
from market_regime.range_detector    import detect_range
from market_regime.volatility_state  import classify_volatility_state
from market_regime.regime_classifier import classify_regime

logger = logging.getLogger(__name__)

# ATR lookback periods for rolling mean comparison
_ATR_HISTORY_PERIOD = 50


class MarketRegimeEngine:
    """Classify the current market regime from multi-engine signal aggregation.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"TQQQ"``).
    timeframe:
        Human-readable label embedded in each output dict (e.g. ``"daily"``).
    slope_window:
        Bars used for the close-price slope regression in trend detection.
    range_lookback:
        Bars used for range high/low measurement.
    sweep_lookback:
        Bars to scan for liquidity sweeps (accum/distrib bias).
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "daily",
        slope_window: int = 30,
        range_lookback: int = 20,
        sweep_lookback: int = 10,
    ) -> None:
        self.symbol         = symbol
        self.timeframe      = timeframe
        self.slope_window   = slope_window
        self.range_lookback = range_lookback
        self.sweep_lookback = sweep_lookback

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute and return the current market regime.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            Regime dict (see module-level docstring for schema).
        """
        if len(df) < 20:
            logger.warning(
                "MarketRegimeEngine[%s]: insufficient data (%d bars, need ≥20)",
                self.symbol, len(df),
            )
            return self._default_result()

        # ── 1. Swing structure from structure_engine ─────────────────────────
        swings      = self._get_swings(df)
        swing_highs = swings.get("swing_highs", [])
        swing_lows  = swings.get("swing_lows",  [])

        # ── 2. ATR data from volatility_engine ──────────────────────────────
        current_atr, atr_history = self._get_atr_data(df)

        # ── 3. Liquidity sweeps from liquidity_map ───────────────────────────
        swept_zones = self._get_swept_zones(df)

        # ── 4. Sub-module classifications ────────────────────────────────────
        trend_result = detect_trend(
            df, swing_highs, swing_lows,
            slope_window=self.slope_window,
        )
        range_result = detect_range(
            df,
            lookback=self.range_lookback,
        )
        vol_state = classify_volatility_state(
            current_atr=current_atr,
            atr_history=atr_history,
        )

        # ── 5. Final regime classification ───────────────────────────────────
        regime_result = classify_regime(
            trend=trend_result,
            range_info=range_result,
            vol_state=vol_state,
            swept_zones=swept_zones,
        )

        result = {
            "symbol":    self.symbol,
            "timeframe": self.timeframe,
            **regime_result,
            "atr_ratio": vol_state.get("atr_ratio", 1.0),
        }

        logger.info(
            "MarketRegimeEngine[%s] → regime=%s confidence=%.2f vol=%s",
            self.symbol,
            result["regime"],
            result["confidence"],
            result["volatility_state"],
        )
        return result

    # ── Engine Adapters (consume — do not reimplement) ─────────────────────────

    def _get_swings(self, df: pd.DataFrame) -> dict:
        """Consume ATR swing detection from structure_engine.swing_detector."""
        try:
            from structure_engine.swing_detector import (
                compute_adaptive_minimum_move,
                detect_swings,
            )
            mm = compute_adaptive_minimum_move(df)
            return detect_swings(df, minimum_move=mm)
        except Exception as exc:
            logger.warning(
                "MarketRegimeEngine[%s]: swing_detector unavailable: %s",
                self.symbol, exc,
            )
            return {"swing_highs": [], "swing_lows": []}

    def _get_atr_data(self, df: pd.DataFrame) -> tuple[float, list[float]]:
        """Consume ATR values from volatility_engine.

        Returns (current_atr, atr_history_list).
        Falls back to a local True Range mean if the engine is unavailable.
        """
        try:
            from volatility_engine.regime import VolatilityRegimeEngine
            eng  = VolatilityRegimeEngine()
            data = eng.get_latest_regime(self.symbol)
            if data:
                current_atr = float(data.get("current_atr", 0.0))
                history     = data.get("atr_history", [])
                if current_atr > 0:
                    return current_atr, history
        except Exception as exc:
            logger.warning(
                "MarketRegimeEngine[%s]: volatility_engine unavailable: %s",
                self.symbol, exc,
            )

        # Fallback: compute local True Range mean from OHLCV
        return self._local_atr_fallback(df)

    def _get_swept_zones(self, df: pd.DataFrame) -> list[dict]:
        """Consume recently swept zones from liquidity_map."""
        try:
            from liquidity_map.sweep_detection import scan_all_zone_sweeps
            from liquidity_map import LiquidityMapEngine

            lme    = LiquidityMapEngine(symbol=self.symbol)
            result = lme.run(df)
            return result.get("swept_zones", [])
        except Exception as exc:
            logger.warning(
                "MarketRegimeEngine[%s]: liquidity_map unavailable: %s",
                self.symbol, exc,
            )
            return []

    def _local_atr_fallback(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> tuple[float, list[float]]:
        """Local True Range mean — only used when volatility_engine is unavailable."""
        trs: list[float] = []
        for i in range(1, len(df)):
            h      = float(df.iloc[i]["high"])
            lo     = float(df.iloc[i]["low"])
            c_prev = float(df.iloc[i - 1]["close"])
            trs.append(max(h - lo, abs(h - c_prev), abs(lo - c_prev)))

        if not trs:
            return 0.0, []

        # Rolling 14-bar ATR via simple mean (Wilder's smoothing is in volatility_engine)
        atr_values: list[float] = []
        for i in range(period - 1, len(trs)):
            atr_values.append(sum(trs[i - period + 1:i + 1]) / period)

        current_atr = atr_values[-1] if atr_values else sum(trs) / len(trs)
        history     = atr_values[-_ATR_HISTORY_PERIOD:]
        return current_atr, history

    # ── Default ────────────────────────────────────────────────────────────────

    def _default_result(self) -> dict[str, Any]:
        return {
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "regime":           "RANGE",
            "confidence":       0.0,
            "trend":            "NEUTRAL",
            "trend_strength":   0.0,
            "volatility_state": "NORMAL",
            "is_range":         True,
            "range_high":       0.0,
            "range_low":        0.0,
            "atr_ratio":        1.0,
        }
