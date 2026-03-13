"""
RoboAlgo — Gann Projection Engine
Calculates Gann angles, time cycles, and price/time symmetry targets.

Gann Theory:
  - Price and time are unified — price should move equal units per equal time units
  - 1×1 angle = 45° = 1 price unit per 1 time unit (most important)
  - 2×1 angle = 1 price unit per 2 time bars (steeper)
  - 1×2 angle = 2 price units per 1 bar (shallower)

Time Cycles (common):
  - 90 calendar days / ~65 trading days
  - 180 calendar days / ~130 trading days
  - 360 calendar days / ~260 trading days (annual cycle)

Price/Time Symmetry:
  - Major moves often repeat in time or price from significant pivots
  - Squaring: price range = time duration (measured in appropriate units)
"""

import logging
from typing import Optional
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sqlalchemy import select, desc

from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# ── Gann Angle Ratios (price per bar) ─────────────────────────────────────────
GANN_ANGLES = {
    "4x1": 4.0,    # very steep
    "3x1": 3.0,    # steep
    "2x1": 2.0,    # fast
    "1x1": 1.0,    # primary (45°)
    "1x2": 0.5,    # slow
    "1x3": 0.333,  # very slow
    "1x4": 0.25,   # sideways
}

# Major time cycle lengths in trading days
TIME_CYCLES = {
    "quarter":   65,   # ~90 calendar days
    "half_year": 130,  # ~180 calendar days
    "annual":    260,  # ~365 calendar days
    "minor":     20,   # monthly cycle
    "weekly":    5,    # weekly cycle
}

PIVOT_ORDER = 5   # swing detection window


class GannEngine:
    """
    Computes Gann-based price targets and time windows.

    This is a pure computation engine — returns live projections
    without storing to DB (Gann projections are computed on-demand per signal).

    Usage:
        engine = GannEngine()
        result = engine.analyze(symbol="SOXL")
        result = engine.project_from_pivot(X=52.30, pivot_bar=100, current_bar=110, price_unit=1.0)
    """

    def analyze(self, symbol: str) -> dict:
        """
        Full Gann analysis for a symbol using DB price history.
        Returns price targets, time windows, and confluence zones.
        """
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return {"error": f"Symbol {symbol} not found"}

            price_rows = session.execute(
                select(PriceData)
                .where(PriceData.instrument_id == instr.id)
                .order_by(PriceData.date)
            ).scalars().all()

        if len(price_rows) < 60:
            return {"symbol": symbol, "error": "Insufficient price history"}

        prices = pd.DataFrame([{
            "date": r.date, "high": r.high, "low": r.low, "close": r.close,
        } for r in price_rows])

        return self._compute_gann_analysis(symbol, prices)

    def compute_gann_score(self, symbol: str) -> float:
        """Return 0–100 score indicating Gann confluence strength."""
        result = self.analyze(symbol)
        if "error" in result or not result.get("angle_confluence"):
            return 50.0

        # Score based on number of angle confluences and cycle alignment
        angle_score   = min(len(result.get("angle_targets", [])) * 10, 40)
        cycle_score   = 30.0 if result.get("in_time_cycle_window") else 10.0
        symmetry_score = result.get("price_time_symmetry_score", 0.0) * 30
        return float(np.clip(angle_score + cycle_score + symmetry_score, 0, 100))

    def project_from_pivot(
        self,
        pivot_price: float,
        pivot_bar: int,
        current_bar: int,
        price_unit: float,
        direction: str = "up"
    ) -> dict:
        """
        Project price targets from a swing pivot using all Gann angles.

        Args:
            pivot_price: Price at the pivot (swing high or low)
            pivot_bar:   Bar index of the pivot
            current_bar: Current bar index
            price_unit:  Price units per bar (ATR or historical range / time span)
            direction:   "up" (bullish) or "down" (bearish)
        """
        bars_elapsed = current_bar - pivot_bar
        targets = {}

        for angle_name, ratio in GANN_ANGLES.items():
            price_change = bars_elapsed * price_unit * ratio
            if direction == "up":
                target = pivot_price + price_change
            else:
                target = pivot_price - price_change
            targets[angle_name] = round(float(target), 4)

        return {
            "pivot_price":    round(pivot_price, 4),
            "bars_elapsed":   bars_elapsed,
            "price_unit":     round(price_unit, 4),
            "direction":      direction,
            "angle_targets":  targets,
        }

    # ── Internal Analysis ──────────────────────────────────────────────────────

    def _compute_gann_analysis(self, symbol: str, prices: pd.DataFrame) -> dict:
        """Full Gann analysis from price history."""
        close = prices["close"].values
        high  = prices["high"].values
        low   = prices["low"].values
        dates = prices["date"].tolist()

        current_price = float(close[-1])
        current_date  = dates[-1]
        n_bars        = len(prices)

        # Find major swing pivots
        swing_hi_idx = argrelextrema(high, np.greater, order=PIVOT_ORDER)[0]
        swing_lo_idx = argrelextrema(low,  np.less,   order=PIVOT_ORDER)[0]

        if len(swing_hi_idx) == 0 or len(swing_lo_idx) == 0:
            return {"symbol": symbol, "error": "No swing pivots found"}

        # Most recent major pivot (the reference point)
        last_hi_idx = int(swing_hi_idx[-1])
        last_lo_idx = int(swing_lo_idx[-1])

        # Determine which pivot is more recent and relevant
        if last_hi_idx > last_lo_idx:
            pivot_type  = "high"
            pivot_price = float(high[last_hi_idx])
            pivot_bar   = last_hi_idx
            direction   = "down"   # from swing high → looking for support
        else:
            pivot_type  = "low"
            pivot_price = float(low[last_lo_idx])
            pivot_bar   = last_lo_idx
            direction   = "up"    # from swing low → looking for resistance

        # Price unit: ATR or historical range / time span
        price_unit = self._compute_price_unit(high, low, close, pivot_bar, n_bars)

        # Gann angle projections from the pivot
        projections = self.project_from_pivot(
            pivot_price, pivot_bar, n_bars - 1, price_unit, direction
        )

        # Time cycle windows
        time_windows = self._compute_time_windows(pivot_bar, n_bars, dates)

        # Price/time symmetry (squaring)
        symmetry = self._compute_price_time_symmetry(
            close, high, low, pivot_bar, pivot_price, price_unit
        )

        # Confluence zones (price levels hit by multiple angles)
        confluence_targets = self._find_angle_confluence(
            projections["angle_targets"], current_price, direction
        )

        # Are we currently in a time cycle window?
        in_cycle_window = any(
            tw["is_active"] for tw in time_windows.values()
        )

        return {
            "symbol":           symbol,
            "current_price":    current_price,
            "current_date":     str(current_date),
            "pivot_type":       pivot_type,
            "pivot_price":      pivot_price,
            "pivot_date":       str(dates[pivot_bar]) if pivot_bar < len(dates) else None,
            "bars_from_pivot":  n_bars - 1 - pivot_bar,
            "price_unit":       price_unit,
            "direction":        direction,
            "angle_targets":    projections["angle_targets"],
            "confluence_targets": confluence_targets,
            "time_windows":     time_windows,
            "in_time_cycle_window": in_cycle_window,
            "price_time_symmetry": symmetry,
            "price_time_symmetry_score": symmetry.get("symmetry_score", 0.0),
            "angle_confluence": len(confluence_targets) > 0,
            "gann_score":       self._score_from_analysis(
                confluence_targets, in_cycle_window, symmetry
            ),
        }

    def _compute_price_unit(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
        pivot_bar: int, current_bar: int
    ) -> float:
        """
        Compute the base price unit for Gann angle scaling.
        Uses ATR as the natural price-per-bar unit.
        """
        # True Range ATR over the swing period
        bars_in_swing = max(current_bar - pivot_bar, 5)
        slice_high  = high[pivot_bar:current_bar + 1]
        slice_low   = low[pivot_bar:current_bar + 1]
        slice_close = close[pivot_bar:current_bar + 1]

        if len(slice_high) < 2:
            return float(abs(close[-1] - close[0]) / max(len(close) - 1, 1))

        tr = np.maximum(
            slice_high[1:] - slice_low[1:],
            np.maximum(
                np.abs(slice_high[1:] - slice_close[:-1]),
                np.abs(slice_low[1:]  - slice_close[:-1])
            )
        )
        return float(np.mean(tr)) if len(tr) > 0 else 0.01

    def _compute_time_windows(
        self, pivot_bar: int, current_bar: int, dates: list
    ) -> dict:
        """
        Determine which Gann time cycles are approaching or active.
        Returns dict of cycle_name → {target_bar, target_date, is_active, bars_to_window}.
        """
        windows = {}
        tolerance = 5  # ± bars = "active" window

        for cycle_name, cycle_bars in TIME_CYCLES.items():
            # Next cycle hit from pivot
            next_bar = pivot_bar + cycle_bars
            bars_to  = next_bar - current_bar
            is_active = abs(bars_to) <= tolerance

            target_date = None
            if next_bar < len(dates):
                target_date = str(dates[next_bar])

            windows[cycle_name] = {
                "cycle_bars":       cycle_bars,
                "target_bar":       next_bar,
                "target_date":      target_date,
                "bars_to_window":   bars_to,
                "is_active":        is_active,
            }

        return windows

    def _compute_price_time_symmetry(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        pivot_bar: int, pivot_price: float, price_unit: float
    ) -> dict:
        """
        Price/Time Symmetry: the current move's price range equals its time span
        when normalized by price_unit. This is "squaring the range."
        """
        bars_elapsed = len(close) - 1 - pivot_bar
        price_range  = abs(close[-1] - pivot_price)
        time_in_price = bars_elapsed * price_unit  # time converted to price units

        if time_in_price == 0:
            return {"symmetry_score": 0.0, "is_squared": False}

        symmetry_ratio = price_range / time_in_price
        is_squared     = 0.8 <= symmetry_ratio <= 1.2  # within ±20% of perfect square

        # Score: perfect square = 1.0, diminishes with distance
        symmetry_score = max(0, 1 - abs(symmetry_ratio - 1.0) * 2)

        return {
            "bars_elapsed":    bars_elapsed,
            "price_range":     round(float(price_range), 4),
            "time_in_price":   round(float(time_in_price), 4),
            "symmetry_ratio":  round(float(symmetry_ratio), 4),
            "is_squared":      is_squared,
            "symmetry_score":  round(float(symmetry_score), 4),
        }

    def _find_angle_confluence(
        self, angle_targets: dict, current_price: float, direction: str,
        confluence_pct: float = 0.02
    ) -> list[dict]:
        """
        Find price levels where multiple Gann angles converge (±2% cluster).
        These are strongest support/resistance zones.
        """
        prices = list(angle_targets.values())
        angles = list(angle_targets.keys())

        confluence = []
        used = set()

        for i in range(len(prices)):
            if i in used:
                continue
            cluster_prices = [prices[i]]
            cluster_angles = [angles[i]]

            for j in range(i + 1, len(prices)):
                if j in used:
                    continue
                if abs(prices[i] - prices[j]) / max(abs(prices[i]), 1e-8) <= confluence_pct:
                    cluster_prices.append(prices[j])
                    cluster_angles.append(angles[j])
                    used.add(j)

            if len(cluster_angles) >= 2:
                cluster_price = float(np.mean(cluster_prices))
                # Only include targets that are ahead of current price in the right direction
                if direction == "up" and cluster_price > current_price:
                    confluence.append({
                        "price":   round(cluster_price, 4),
                        "angles":  cluster_angles,
                        "strength": len(cluster_angles),
                    })
                elif direction == "down" and cluster_price < current_price:
                    confluence.append({
                        "price":   round(cluster_price, 4),
                        "angles":  cluster_angles,
                        "strength": len(cluster_angles),
                    })
            used.add(i)

        return sorted(confluence, key=lambda x: x["strength"], reverse=True)

    def _score_from_analysis(
        self,
        confluence_targets: list[dict],
        in_cycle_window: bool,
        symmetry: dict
    ) -> float:
        """Compute 0–100 Gann score for confluence engine."""
        score = 30.0  # base

        # Angle confluence zones
        if confluence_targets:
            score += min(len(confluence_targets) * 10, 30)
            # Strong clusters (3+ angles)
            strong = any(t["strength"] >= 3 for t in confluence_targets)
            if strong:
                score += 10

        # Time cycle alignment
        if in_cycle_window:
            score += 20

        # Price/time symmetry (squaring)
        sym_score = symmetry.get("symmetry_score", 0)
        score += sym_score * 10

        return float(np.clip(score, 0, 100))
