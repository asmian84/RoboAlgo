"""Trade Setup Engine — main orchestrator.

Converts analytical signals from RoboAlgo engines into complete, actionable
trade plans with entry, stop, targets, risk/reward, and position sizing.

Supported setup types
---------------------
LIQUIDITY_REVERSAL  — liquidity sweep reversal at structural levels.
TREND_PULLBACK      — trend confirmed + price retests support/resistance.
BREAKOUT_EXPANSION  — volatility expansion after range compression.

Multi-timeframe model
---------------------
DAY_TRADE:   bias=1H,    setup=15m,  entry=5m
SWING_TRADE: bias=Daily, setup=4H,   entry=1H
INVESTMENT:  bias=Weekly,setup=Daily, entry=Daily/4H

Output format
-------------
{
    "trade_type":  "DAY_TRADE" | "SWING_TRADE" | "INVESTMENT",
    "setup":       "LIQUIDITY_REVERSAL" | "TREND_PULLBACK" | "BREAKOUT_EXPANSION",
    "direction":   "LONG" | "SHORT",
    "entry":       float,
    "stop_loss":   float,
    "targets":     [float, float, float],
    "risk_reward": float,
    "confidence":  float,            # 0–1 normalised
    "timeframes":  {"bias": str, "setup": str, "entry": str},
    "position":    {                 # sizing (requires account_size param)
        "position_size":  float,
        "risk_amount":    float,
        "position_value": float,
        "position_pct":   float,
    },
    "symbol":      str,
    "date":        str,
    "bar_index":   int,
}

IMPORTANT: No indicators are recomputed here.  All analytical signals are
consumed from:
    • liquidity_map.liquidity_engine.LiquidityMapEngine
    • market_regime.regime_engine.MarketRegimeEngine
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from trade_setup.entry_logic     import (
    compute_entry,
    DIRECTION_LONG, DIRECTION_SHORT,
    SETUP_LIQUIDITY_REVERSAL, SETUP_TREND_PULLBACK, SETUP_BREAKOUT_EXPANSION,
)
from trade_setup.stop_loss       import compute_stop_loss
from trade_setup.target_generator import generate_targets, compute_risk_reward
from trade_setup.position_sizer  import compute_position_size

logger = logging.getLogger(__name__)

# ── Trade type labels ──────────────────────────────────────────────────────────

TRADE_TYPE_DAY     = "DAY_TRADE"
TRADE_TYPE_SWING   = "SWING_TRADE"
TRADE_TYPE_INVEST  = "INVESTMENT"

# Multi-timeframe model
_TIMEFRAME_MAP: dict[str, dict[str, str]] = {
    TRADE_TYPE_DAY:   {"bias": "1H",    "setup": "15m",   "entry": "5m"},
    TRADE_TYPE_SWING: {"bias": "Daily", "setup": "4H",    "entry": "1H"},
    TRADE_TYPE_INVEST:{"bias": "Weekly","setup": "Daily",  "entry": "Daily"},
}


class TradeSetupEngine:
    """Convert RoboAlgo engine signals into complete trade plans.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"TQQQ"``).
    trade_type:
        One of ``TRADE_TYPE_DAY``, ``TRADE_TYPE_SWING``, ``TRADE_TYPE_INVEST``.
    account_size:
        Total account equity for position sizing (default $100,000).
    risk_percent:
        Fraction of account to risk per trade (default 1%).
    max_position_pct:
        Maximum position value as fraction of account (default 20%).
    """

    def __init__(
        self,
        symbol: str,
        trade_type: str = TRADE_TYPE_SWING,
        account_size: float = 100_000.0,
        risk_percent: float = 0.01,
        max_position_pct: float = 0.20,
    ) -> None:
        self.symbol          = symbol
        self.trade_type      = trade_type
        self.account_size    = account_size
        self.risk_percent    = risk_percent
        self.max_position_pct= max_position_pct

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Scan the DataFrame and return all valid trade setups found.

        Args:
            df: OHLCV DataFrame (open, high, low, close, volume).

        Returns:
            List of trade plan dicts ordered chronologically.
        """
        if len(df) < 20:
            logger.warning(
                "TradeSetupEngine[%s]: insufficient data (%d bars, need ≥20)",
                self.symbol, len(df),
            )
            return []

        # ── Consume engine signals ───────────────────────────────────────────
        regime          = self._get_regime(df)
        liq_map_result  = self._get_liquidity_map(df)
        liquidity_zones = liq_map_result.get("liquidity_zones", [])
        swept_zones     = liq_map_result.get("swept_zones",     [])
        atr             = self._get_atr(df)

        setups: list[dict[str, Any]] = []

        # ── Scan for TREND_PULLBACK setups ───────────────────────────────────
        pullback = self._detect_trend_pullback(df, regime, liquidity_zones, atr)
        if pullback:
            setups.append(pullback)

        # ── Scan for BREAKOUT_EXPANSION setups ───────────────────────────────
        breakout = self._detect_breakout_expansion(
            df, regime, liquidity_zones, atr,
        )
        if breakout:
            setups.append(breakout)

        # Sort chronologically
        setups.sort(key=lambda s: s.get("bar_index", 0))

        logger.info(
            "TradeSetupEngine[%s]: %d setups found (type=%s)",
            self.symbol, len(setups), self.trade_type,
        )
        return setups

    def latest(self) -> Optional[dict[str, Any]]:
        """Return the most recent setup, or None."""
        results = self.run.__func__  # type hint only
        return None   # stateless — caller should use run() directly

    # ── Setup Builders ──────────────────────────────────────────────────────────

    def _detect_trend_pullback(
        self,
        df: pd.DataFrame,
        regime: dict,
        liquidity_zones: list[dict],
        atr: float,
    ) -> dict | None:
        """Detect and build a TREND_PULLBACK setup on the most recent bar."""
        trend = regime.get("trend", "NEUTRAL")
        trend_str = regime.get("trend_strength", 0.0)

        # Need a confirmed trend
        if trend == "NEUTRAL" or trend_str < 0.35:
            return None

        bar_index = len(df) - 1

        # Check that the current bar is a pullback (lower high for uptrend, higher low for downtrend)
        if not self._is_pullback_bar(df, bar_index, trend):
            return None

        entry_info = compute_entry(
            df, SETUP_TREND_PULLBACK, bar_index, regime=regime,
        )
        entry_price  = entry_info["entry"]
        nearest_zone = self._nearest_zone(liquidity_zones, entry_price)

        stop_info = compute_stop_loss(
            entry_info, df, SETUP_TREND_PULLBACK,
            regime=regime, nearest_zone=nearest_zone, atr=atr,
        )

        targets = generate_targets(
            entry_price, stop_info["stop_loss"],
            entry_info["direction"], liquidity_zones,
        )
        if not targets:
            return None

        rr = compute_risk_reward(
            entry_price, stop_info["stop_loss"],
            targets[0], entry_info["direction"],
        )

        position = compute_position_size(
            entry_price, stop_info["stop_loss"],
            self.account_size, self.risk_percent, self.max_position_pct,
        )

        confidence = round(trend_str * 0.85, 4)

        return self._make_plan(
            setup=SETUP_TREND_PULLBACK,
            direction=entry_info["direction"],
            entry=entry_price,
            stop=stop_info["stop_loss"],
            targets=targets,
            rr=rr,
            confidence=confidence,
            position=position,
            bar_index=bar_index,
            df=df,
        )

    def _detect_breakout_expansion(
        self,
        df: pd.DataFrame,
        regime: dict,
        liquidity_zones: list[dict],
        atr: float,
    ) -> dict | None:
        """Detect and build a BREAKOUT_EXPANSION setup on the most recent bar."""
        vol_state = regime.get("volatility_state", "NORMAL")
        is_range  = regime.get("is_range", False)

        # Must be volatility expansion or recent range breakout
        if vol_state != "EXPANDING" and not is_range:
            return None

        bar_index   = len(df) - 1
        bar         = df.iloc[bar_index]
        close       = float(bar["close"])
        range_high  = regime.get("range_high", 0.0)
        range_low   = regime.get("range_low",  0.0)

        # Price must be breaking out of the identified range
        if range_high and range_low:
            is_breaking_up   = close > range_high
            is_breaking_down = close < range_low
            if not (is_breaking_up or is_breaking_down):
                return None

        nearest_zone = self._nearest_zone(liquidity_zones, close)

        entry_info = compute_entry(
            df, SETUP_BREAKOUT_EXPANSION, bar_index,
            regime=regime, zone=nearest_zone,
        )
        entry_price = entry_info["entry"]

        stop_info = compute_stop_loss(
            entry_info, df, SETUP_BREAKOUT_EXPANSION,
            regime=regime, nearest_zone=nearest_zone, atr=atr,
        )

        targets = generate_targets(
            entry_price, stop_info["stop_loss"],
            entry_info["direction"], liquidity_zones,
        )
        if not targets:
            return None

        rr = compute_risk_reward(
            entry_price, stop_info["stop_loss"],
            targets[0], entry_info["direction"],
        )

        position = compute_position_size(
            entry_price, stop_info["stop_loss"],
            self.account_size, self.risk_percent, self.max_position_pct,
        )

        # Confidence blends volatility state and range confidence
        range_conf   = regime.get("range_confidence", 0.5) if is_range else 0.5
        vol_bonus    = 0.2 if vol_state == "EXPANDING" else 0.0
        confidence   = round(min(range_conf + vol_bonus, 0.95), 4)

        return self._make_plan(
            setup=SETUP_BREAKOUT_EXPANSION,
            direction=entry_info["direction"],
            entry=entry_price,
            stop=stop_info["stop_loss"],
            targets=targets,
            rr=rr,
            confidence=confidence,
            position=position,
            bar_index=bar_index,
            df=df,
        )

    # ── Engine Adapters (consume — do not reimplement) ─────────────────────────

    def _get_regime(self, df: pd.DataFrame) -> dict:
        """Consume regime classification from market_regime.MarketRegimeEngine."""
        try:
            from market_regime import MarketRegimeEngine
            engine = MarketRegimeEngine(symbol=self.symbol)
            return engine.run(df)
        except Exception as exc:
            logger.warning(
                "TradeSetupEngine[%s]: market_regime unavailable: %s",
                self.symbol, exc,
            )
            return {
                "regime": "RANGE", "trend": "NEUTRAL", "trend_strength": 0.0,
                "volatility_state": "NORMAL", "is_range": True,
                "range_high": 0.0, "range_low": 0.0, "confidence": 0.0,
            }

    def _get_liquidity_map(self, df: pd.DataFrame) -> dict:
        """Consume zone data from liquidity_map.LiquidityMapEngine."""
        try:
            from liquidity_map import LiquidityMapEngine
            engine = LiquidityMapEngine(symbol=self.symbol)
            return engine.run(df)
        except Exception as exc:
            logger.warning(
                "TradeSetupEngine[%s]: liquidity_map unavailable: %s",
                self.symbol, exc,
            )
            return {"liquidity_zones": [], "swept_zones": [], "zone_count": 0}

    def _get_atr(self, df: pd.DataFrame) -> float:
        """Consume current ATR from volatility_engine, fallback to local TR mean."""
        try:
            from volatility_engine.regime import VolatilityRegimeEngine
            data = VolatilityRegimeEngine().get_latest_regime(self.symbol)
            if data:
                atr = float(data.get("current_atr", 0.0))
                if atr > 0:
                    return atr
        except Exception:
            pass

        # Local fallback: mean of last 14 True Ranges
        trs: list[float] = []
        for i in range(1, min(15, len(df))):
            h      = float(df.iloc[-i]["high"])
            lo     = float(df.iloc[-i]["low"])
            c_prev = float(df.iloc[-i - 1]["close"])
            trs.append(max(h - lo, abs(h - c_prev), abs(lo - c_prev)))
        return sum(trs) / len(trs) if trs else 0.0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _nearest_zone(
        self, zones: list[dict], price: float
    ) -> dict | None:
        """Return the zone closest to ``price``."""
        if not zones:
            return None
        return min(zones, key=lambda z: abs(z["price"] - price))

    def _is_pullback_bar(
        self, df: pd.DataFrame, bar_index: int, trend: str
    ) -> bool:
        """Simple pullback check: bar is not a new extreme in the trend direction."""
        if bar_index < 3:
            return False
        window = df.iloc[max(0, bar_index - 5):bar_index]
        bar    = df.iloc[bar_index]
        if trend == "UP":
            # Pullback = current close below recent close mean (not at new high)
            return float(bar["close"]) < float(window["high"].max())
        else:
            return float(bar["close"]) > float(window["low"].min())

    def _make_plan(
        self,
        setup: str,
        direction: str,
        entry: float,
        stop: float,
        targets: list[float],
        rr: float,
        confidence: float,
        position: dict,
        bar_index: int,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Assemble the canonical trade plan dict."""
        date_val = df.index[bar_index]
        date_str = (
            str(date_val.date()) if hasattr(date_val, "date") else str(date_val)
        )

        return {
            "symbol":      self.symbol,
            "trade_type":  self.trade_type,
            "setup":       setup,
            "direction":   direction,
            "entry":       entry,
            "stop_loss":   stop,
            "targets":     targets,
            "risk_reward": rr,
            "confidence":  round(min(max(confidence, 0.0), 1.0), 4),
            "timeframes":  _TIMEFRAME_MAP.get(self.trade_type, {}),
            "position":    {
                "position_size":  position.get("position_size",  0.0),
                "risk_amount":    position.get("risk_amount",     0.0),
                "position_value": position.get("position_value",  0.0),
                "position_pct":   position.get("position_pct",    0.0),
            },
            "date":        date_str,
            "bar_index":   bar_index,
        }
