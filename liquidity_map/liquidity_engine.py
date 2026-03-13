"""Liquidity Map Engine — main orchestrator.

Aggregates all liquidity zone types into a unified map and tracks which
zones have been swept (stop-hunted) by recent price action.

Engine responsibilities
-----------------------
1. Consume ATR-adaptive swing highs/lows from structure_engine.swing_detector.
2. Detect equal high/low clusters via liquidity_map.equal_levels.
3. Build the full zone list via liquidity_map.liquidity_zones.build_zones.
4. Identify swept zones via liquidity_map.sweep_detection.scan_all_zone_sweeps.
5. Expose proximity filtering (zones near current price).

Output format
-------------
{
    "symbol":         str,
    "zone_count":     int,
    "liquidity_zones": [
        {
            "type":     str,    # zone type identifier
            "price":    float,
            "strength": float,  # 0–1
            "side":     str,    # "high" | "low"
            "meta":     dict,
        },
        ...
    ],
    "swept_zones": [
        {
            "zone":       dict,  # the zone that was swept
            "bar_index":  int,
            "sweep_type": str,   # "top_sweep" | "bottom_sweep"
            "date":       str,
        },
        ...
    ],
    "strongest_zone": dict | None,   # highest-strength unswept zone
}

IMPORTANT: No indicators are recomputed here.  Swing detection is consumed
from structure_engine.swing_detector via the adapter below.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from liquidity_map.equal_levels import detect_equal_highs, detect_equal_lows
from liquidity_map.liquidity_zones import build_zones
from liquidity_map.sweep_detection import scan_all_zone_sweeps

logger = logging.getLogger(__name__)


class LiquidityMapEngine:
    """Build and maintain a liquidity zone map for a single symbol.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"TQQQ"``).
    equal_tolerance:
        Relative price tolerance for equal high/low detection (default 0.2%).
    equal_min_touches:
        Minimum touches to form an equal level cluster (default 2).
    equal_lookback:
        Bar window for equal level scanning (default 50).
    range_lookback:
        Bar window used to compute the consolidation range (default 20).
    sweep_lookback:
        Number of recent bars to scan for zone sweeps (default 5).
    """

    def __init__(
        self,
        symbol: str,
        equal_tolerance: float = 0.002,
        equal_min_touches: int = 2,
        equal_lookback: int = 50,
        range_lookback: int = 20,
        sweep_lookback: int = 5,
    ) -> None:
        self.symbol            = symbol
        self.equal_tolerance   = equal_tolerance
        self.equal_min_touches = equal_min_touches
        self.equal_lookback    = equal_lookback
        self.range_lookback    = range_lookback
        self.sweep_lookback    = sweep_lookback

        self._zones:        list[dict] = []
        self._swept_zones:  list[dict] = []

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Build the full liquidity map for the supplied OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.
                Index may be datetime or integer.

        Returns:
            Liquidity map dict (see module-level docstring for schema).
        """
        if len(df) < 10:
            logger.warning(
                "LiquidityMapEngine[%s]: insufficient data (%d bars, need ≥10)",
                self.symbol, len(df),
            )
            return self._empty_result()

        # ── 1. Swing highs / lows from structure_engine ──────────────────────
        swings       = self._get_swings(df)
        swing_highs  = swings.get("swing_highs", [])
        swing_lows   = swings.get("swing_lows",  [])

        # ── 2. Equal level clusters ──────────────────────────────────────────
        eq_highs = detect_equal_highs(
            df,
            tolerance=self.equal_tolerance,
            min_touches=self.equal_min_touches,
            lookback=self.equal_lookback,
        )
        eq_lows = detect_equal_lows(
            df,
            tolerance=self.equal_tolerance,
            min_touches=self.equal_min_touches,
            lookback=self.equal_lookback,
        )

        # ── 3. Build unified zone list ───────────────────────────────────────
        self._zones = build_zones(
            df=df,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            equal_high_clusters=eq_highs,
            equal_low_clusters=eq_lows,
            range_lookback=self.range_lookback,
        )

        # ── 4. Detect swept zones ────────────────────────────────────────────
        self._swept_zones = scan_all_zone_sweeps(
            df,
            zones=self._zones,
            lookback=self.sweep_lookback,
        )

        # ── 5. Identify the strongest unswept zone ───────────────────────────
        swept_prices = {e["zone"]["price"] for e in self._swept_zones}
        unswept = [z for z in self._zones if z["price"] not in swept_prices]
        strongest = (
            max(unswept, key=lambda z: z["strength"]) if unswept else None
        )

        logger.info(
            "LiquidityMapEngine[%s]: %d zones built, %d swept",
            self.symbol, len(self._zones), len(self._swept_zones),
        )

        return {
            "symbol":          self.symbol,
            "zone_count":      len(self._zones),
            "liquidity_zones": self._zones,
            "swept_zones":     self._swept_zones,
            "strongest_zone":  strongest,
        }

    def get_zones_near_price(
        self,
        current_price: float,
        proximity_pct: float = 0.02,
    ) -> list[dict]:
        """Return zones within ``proximity_pct`` of ``current_price``.

        Args:
            current_price: The reference price to measure proximity from.
            proximity_pct: Maximum relative distance to include a zone
                           (e.g. 0.02 = within ±2%).  Default 2%.

        Returns:
            Filtered list of zone dicts, sorted by proximity (nearest first).
        """
        if not self._zones:
            return []

        near = [
            z for z in self._zones
            if abs(z["price"] - current_price) / max(current_price, 1e-9) <= proximity_pct
        ]
        near.sort(key=lambda z: abs(z["price"] - current_price))
        return near

    def get_zones(self) -> list[dict]:
        """Return the full zone list from the most recent :meth:`run` call."""
        return self._zones

    def get_swept_zones(self) -> list[dict]:
        """Return swept zone events from the most recent :meth:`run` call."""
        return self._swept_zones

    def reset(self) -> None:
        """Clear cached zone data (call when switching symbols)."""
        self._zones       = []
        self._swept_zones = []

    # ── Engine Adapter ─────────────────────────────────────────────────────────

    def _get_swings(self, df: pd.DataFrame) -> dict:
        """Consume ATR swing detection from structure_engine.swing_detector.

        Degrades gracefully to empty swing lists if the engine is unavailable.
        """
        try:
            from structure_engine.swing_detector import (
                compute_adaptive_minimum_move,
                detect_swings,
            )
            mm = compute_adaptive_minimum_move(df)
            return detect_swings(df, minimum_move=mm)
        except Exception as exc:
            logger.warning(
                "LiquidityMapEngine[%s]: swing_detector unavailable: %s",
                self.symbol, exc,
            )
            return {"swing_highs": [], "swing_lows": []}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _empty_result(self) -> dict[str, Any]:
        return {
            "symbol":          self.symbol,
            "zone_count":      0,
            "liquidity_zones": [],
            "swept_zones":     [],
            "strongest_zone":  None,
        }
