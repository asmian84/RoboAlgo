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
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from market_regime.trend_detector    import detect_trend
from market_regime.range_detector    import detect_range
from market_regime.volatility_state  import classify_volatility_state
from market_regime.regime_classifier import classify_regime

logger = logging.getLogger(__name__)

# ATR lookback periods for rolling mean comparison
_ATR_HISTORY_PERIOD = 50

# NYSE regular session in ET — used to decide when crypto is a *leading*
# indicator (equity market closed) vs. concurrent noise (market open).
_MARKET_OPEN_MIN  = 9 * 60 + 30   # 09:30 ET
_MARKET_CLOSE_MIN = 16 * 60       # 16:00 ET
_ET = ZoneInfo("America/New_York")


def _is_equity_market_open(now: datetime | None = None) -> bool:
    """Regular-session check for NYSE (ignores half-days and holidays).

    Weekends → closed. Weekdays outside 09:30–16:00 ET → closed.
    Half-days and holidays are intentionally ignored — worst case the crypto
    lead is treated as concurrent noise instead of leading signal, a harmless
    downgrade.
    """
    now = now or datetime.now(_ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    else:
        now = now.astimezone(_ET)
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return _MARKET_OPEN_MIN <= m < _MARKET_CLOSE_MIN


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

        # ── 6. Crypto lead overlay (BTC as 24/7 risk-sentiment proxy) ─────
        lead = self._compute_crypto_lead()
        if lead is not None:
            result["crypto_lead"] = lead
            self._apply_crypto_lead(result, lead)

        logger.info(
            "MarketRegimeEngine[%s] → regime=%s confidence=%.2f vol=%s%s",
            self.symbol,
            result["regime"],
            result["confidence"],
            result["volatility_state"],
            f" btc_lead={lead['direction']}({lead['btc_change_pct']:+.2f}%)" if lead else "",
        )
        return result

    def _compute_crypto_lead(self) -> dict[str, Any] | None:
        """BTC's most recent daily move + whether it counts as a leading signal.

        When the equity market is closed (weekend / overnight), BTC's action
        is a real-time risk-sentiment proxy that historically front-runs
        equity opens. When the market is open, we still surface the value for
        transparency but don't treat it as leading signal.

        Returns None if BTC data is unavailable (never raises).
        """
        try:
            from data_engine.crypto_fetcher import fetch_crypto_daily
            btc = fetch_crypto_daily("BTC", bars=5)
            if btc is None or len(btc) < 2:
                return None
            latest = float(btc["close"].iloc[-1])
            prev   = float(btc["close"].iloc[-2])
            pct    = ((latest - prev) / prev * 100.0) if prev else 0.0

            if abs(pct) >= 3.0:
                magnitude = "LARGE"
            elif abs(pct) >= 1.0:
                magnitude = "MEDIUM"
            else:
                magnitude = "SMALL"

            if pct > 0.3:
                direction = "UP"
            elif pct < -0.3:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            market_open = _is_equity_market_open()
            return {
                "btc_change_pct":    round(pct, 2),
                "direction":         direction,
                "magnitude":         magnitude,
                "equity_market_open": market_open,
                # Leading only when equities are closed and BTC moved enough
                # to be meaningful (not intraday noise).
                "leading_indicator": (not market_open) and magnitude in ("MEDIUM", "LARGE"),
            }
        except Exception as exc:
            logger.debug(
                "MarketRegimeEngine[%s]: crypto lead unavailable: %s",
                self.symbol, exc,
            )
            return None

    def _apply_crypto_lead(self, result: dict[str, Any], lead: dict[str, Any]) -> None:
        """Nudge confidence by ±0.05 when BTC leads in / against the trend.

        Applied only when `leading_indicator` is True (i.e. equity market
        is closed AND BTC magnitude is MEDIUM+). Capped so the base signal
        always dominates — this is a modifier, not an override.
        """
        if not lead.get("leading_indicator"):
            return

        trend     = result.get("trend", "NEUTRAL")
        direction = lead["direction"]
        conf      = float(result.get("confidence", 0.0))
        mods: list[str] = result.setdefault("modifiers", [])

        if trend == direction and trend in ("UP", "DOWN"):
            result["confidence"] = min(1.0, conf + 0.05)
            mods.append(f"BTC leading {direction} over closed window (+0.05 conf)")
        elif (trend == "UP" and direction == "DOWN") or (trend == "DOWN" and direction == "UP"):
            result["confidence"] = max(0.0, conf - 0.05)
            mods.append(f"BTC {direction} disagrees with equity {trend} (-0.05 conf)")

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
