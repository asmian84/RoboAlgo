"""
RoboAlgo — Liquidity Map Engine
Identifies where liquidity is likely clustered ahead of price.

Liquidity accumulates around key reference prices where stop orders and pending
orders concentrate.  Understanding these zones lets the system assess whether a
breakout is moving INTO a liquidity pool (high-probability continuation) or
moving AWAY from one (fading momentum).

Score: LiquidityLevelScore 0–100 per detected level.

Level Sources
-------------
* N-bar swing highs and swing lows   (structural pivots)
* Previous-day high / low            (intraday reference)
* Compression range boundaries       (max high / min low in compression window)

Clustering
----------
Nearby levels are merged into a single zone when they are within
CLUSTER_TOLERANCE_PCT (0.3 %) of each other.

LiquidityLevelScore Components
-------------------------------
touch_count   (40%)
    Number of times price approached within APPROACH_PCT of the level.
    Score = min(touches / MAX_TOUCHES, 1.0) × 100.

volume_cluster (40%)
    Sum of volume in bars where price was within APPROACH_PCT of the level,
    normalised against the full-window average daily volume.
    Score = min(vol_ratio / VOL_RATIO_CAP, 1.0) × 100.

proximity     (20%)
    How close is the level to current price?
    Distance < 0.5 % → 100; Distance > 5 % → 0.
    Score = max(0, 100 − distance_pct / PROXIMITY_CAP × 100).

Liquidity Alignment
-------------------
Given a breakout direction (long / short), alignment is high when:
  - Long  breakout + strong liquidity pool ABOVE current price → aligned
  - Short breakout + strong liquidity pool BELOW current price → aligned
Used by SetupQualityScorer as factor `liquidity_alignment_score` (5 % weight).

Output
------
{
  symbol, date,
  liquidity_levels: [
    { price_level, liquidity_score, distance_pct, side, touch_count },
    ...   # sorted by score descending
  ],
  nearest_above: { price_level, liquidity_score, distance_pct } | None,
  nearest_below: { price_level, liquidity_score, distance_pct } | None,
  alignment_score: float          # 0–100 for current breakout direction
  computed_at: ISO timestamp
}
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc

from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SWING_LOOKBACK      = 40     # bars to scan for swing highs/lows
CLUSTER_TOLERANCE   = 0.003  # 0.3 % — merge levels within this radius
APPROACH_PCT        = 0.005  # 0.5 % — consider price "near" a level
MAX_TOUCHES         = 8      # touches that give a 100 touch score
VOL_RATIO_CAP       = 3.0    # volume ratio above this → 100 vol score
PROXIMITY_CAP       = 0.05   # 5 % away → 0 proximity score
MAX_LEVELS          = 10     # maximum levels to return

LEVEL_WEIGHTS = {
    "touch_count":    0.40,
    "volume_cluster": 0.40,
    "proximity":      0.20,
}
assert abs(sum(LEVEL_WEIGHTS.values()) - 1.0) < 1e-9


def _linear_score(value: float, floor: float, cap: float) -> float:
    if value <= floor:
        return 0.0
    if value >= cap:
        return 100.0
    return (value - floor) / (cap - floor) * 100.0


class LiquidityMapEngine:
    """
    Builds a map of liquidity clusters around current price.

    Usage:
        engine = LiquidityMapEngine()
        result = engine.build_liquidity_map("SOXL")

        # For signal scoring:
        alignment = engine.get_alignment_score("SOXL", direction="long")
    """

    def build_liquidity_map(
        self,
        symbol:     str,
        as_of_date: Optional[date] = None,
    ) -> dict:
        """
        Detect and score all liquidity levels for *symbol*.

        Returns the structured map with scored levels sorted by
        liquidity_score descending.
        """
        target_date = as_of_date or date.today()
        needed_bars = SWING_LOOKBACK + 10

        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(
                    Instrument.symbol == symbol.upper()
                )
            ).scalar_one_or_none()
            if inst is None:
                return self._empty(symbol, target_date, error=f"Symbol {symbol} not found")

            prices = pd.read_sql(
                select(
                    PriceData.date, PriceData.high, PriceData.low,
                    PriceData.close, PriceData.volume,
                )
                .where(
                    PriceData.instrument_id == inst.id,
                    PriceData.date <= target_date,
                )
                .order_by(desc(PriceData.date))
                .limit(needed_bars),
                session.bind,
            )

        if prices.empty or len(prices) < 10:
            return self._empty(symbol, target_date, error="Insufficient price data")

        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values("date").reset_index(drop=True)

        current_price = float(prices["close"].iloc[-1])
        avg_volume    = float(prices["volume"].mean()) if "volume" in prices else 1.0

        # ── Step 1: Collect candidate levels ──────────────────────────────────
        candidate_levels = []

        # Swing highs and swing lows (N-bar pivots with at least 2 bars on each side)
        pivot_window = 3
        for i in range(pivot_window, len(prices) - pivot_window):
            bar_high = float(prices["high"].iloc[i])
            bar_low  = float(prices["low"].iloc[i])

            # Swing high: higher than pivot_window bars on both sides
            if all(bar_high >= float(prices["high"].iloc[i - j]) for j in range(1, pivot_window + 1)) and \
               all(bar_high >= float(prices["high"].iloc[i + j]) for j in range(1, pivot_window + 1)):
                candidate_levels.append(bar_high)

            # Swing low: lower than pivot_window bars on both sides
            if all(bar_low <= float(prices["low"].iloc[i - j]) for j in range(1, pivot_window + 1)) and \
               all(bar_low <= float(prices["low"].iloc[i + j]) for j in range(1, pivot_window + 1)):
                candidate_levels.append(bar_low)

        # Previous-day high and low (most recent complete bar)
        if len(prices) >= 2:
            prev_bar = prices.iloc[-2]
            candidate_levels.append(float(prev_bar["high"]))
            candidate_levels.append(float(prev_bar["low"]))

        # All-window high and low (compression boundaries)
        candidate_levels.append(float(prices["high"].max()))
        candidate_levels.append(float(prices["low"].min()))

        if not candidate_levels:
            return self._empty(symbol, target_date, error="No candidate levels found")

        # ── Step 2: Cluster nearby levels ─────────────────────────────────────
        clustered = self._cluster_levels(candidate_levels, current_price, CLUSTER_TOLERANCE)

        # ── Step 3: Score each cluster ────────────────────────────────────────
        scored_levels = []
        for level_price in clustered:
            tolerance_abs = level_price * APPROACH_PCT

            # Bars where price was "near" the level (high or low touched it)
            near_mask = (
                (prices["high"] >= level_price - tolerance_abs) &
                (prices["low"]  <= level_price + tolerance_abs)
            )
            near_bars = prices[near_mask]
            touches   = int(near_mask.sum())

            # Touch score
            touch_score = min(touches / MAX_TOUCHES, 1.0) * 100.0

            # Volume cluster score
            near_volume = float(near_bars["volume"].sum()) if not near_bars.empty else 0.0
            near_days   = max(len(near_bars), 1)
            avg_near_vol = near_volume / near_days
            vol_ratio    = avg_near_vol / max(avg_volume, 1.0)
            vol_score    = min(vol_ratio / VOL_RATIO_CAP, 1.0) * 100.0

            # Proximity score
            distance_pct = abs(current_price - level_price) / max(current_price, 1e-6)
            prox_score   = max(0.0, 100.0 - (distance_pct / PROXIMITY_CAP) * 100.0)
            prox_score   = min(prox_score, 100.0)

            # Weighted total
            liquidity_score = round(
                touch_score  * LEVEL_WEIGHTS["touch_count"]   +
                vol_score    * LEVEL_WEIGHTS["volume_cluster"] +
                prox_score   * LEVEL_WEIGHTS["proximity"],
                1,
            )

            side = "above" if level_price >= current_price else "below"

            scored_levels.append({
                "price_level":     round(level_price, 2),
                "liquidity_score": liquidity_score,
                "distance_pct":    round(distance_pct * 100, 2),
                "side":            side,
                "touch_count":     touches,
                "components": {
                    "touch_count":    round(touch_score, 1),
                    "volume_cluster": round(vol_score, 1),
                    "proximity":      round(prox_score, 1),
                },
            })

        # Sort by score descending, limit to MAX_LEVELS
        scored_levels.sort(key=lambda x: x["liquidity_score"], reverse=True)
        scored_levels = scored_levels[:MAX_LEVELS]

        # ── Nearest above / below ──────────────────────────────────────────────
        above_levels = [l for l in scored_levels if l["side"] == "above"]
        below_levels = [l for l in scored_levels if l["side"] == "below"]

        nearest_above = min(above_levels, key=lambda x: x["distance_pct"], default=None)
        nearest_below = min(below_levels, key=lambda x: x["distance_pct"], default=None)

        return {
            "symbol":          symbol.upper(),
            "date":            target_date.isoformat(),
            "current_price":   round(current_price, 2),
            "liquidity_levels": scored_levels,
            "nearest_above":   nearest_above,
            "nearest_below":   nearest_below,
            "computed_at":     datetime.utcnow().isoformat() + "Z",
        }

    def get_alignment_score(
        self,
        symbol:     str,
        direction:  str = "long",   # "long" | "short"
        as_of_date: Optional[date] = None,
    ) -> float:
        """
        Return a 0–100 alignment score indicating how well the breakout direction
        aligns with nearby liquidity.

        Long  breakout: high score when strong liquidity sits ABOVE price.
        Short breakout: high score when strong liquidity sits BELOW price.

        Returns 0.0 on error.
        """
        try:
            result = self.build_liquidity_map(symbol, as_of_date)
            levels = result.get("liquidity_levels", [])
            if not levels:
                return 0.0

            # Filter to levels on the aligned side
            aligned_side = "above" if direction == "long" else "below"
            aligned_levels = [l for l in levels if l["side"] == aligned_side]

            if not aligned_levels:
                return 0.0

            # Weighted alignment: favour close, high-score levels
            best = max(aligned_levels, key=lambda l: l["liquidity_score"])
            # Discount for distance: level within 2% = full score, 5%+ = halved
            dist_discount = max(0.5, 1.0 - best["distance_pct"] / 10.0)
            return round(best["liquidity_score"] * dist_discount, 1)

        except Exception as e:
            logger.warning("LiquidityMapEngine.get_alignment_score failed for %s: %s", symbol, e)
            return 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_levels(
        levels: list[float],
        reference_price: float,
        tolerance_pct: float,
    ) -> list[float]:
        """
        Merge candidate levels that are within tolerance_pct of each other.
        Returns the centroid of each cluster.
        """
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters: list[list[float]] = []
        current_cluster: list[float] = [sorted_levels[0]]

        for lvl in sorted_levels[1:]:
            last = current_cluster[-1]
            if (lvl - last) / max(last, 1e-6) <= tolerance_pct:
                current_cluster.append(lvl)
            else:
                clusters.append(current_cluster)
                current_cluster = [lvl]
        clusters.append(current_cluster)

        return [float(np.mean(c)) for c in clusters]

    @staticmethod
    def _empty(symbol: str, d: date, error: str = "") -> dict:
        return {
            "symbol":           symbol.upper(),
            "date":             d.isoformat(),
            "current_price":    0.0,
            "liquidity_levels": [],
            "nearest_above":    None,
            "nearest_below":    None,
            "computed_at":      datetime.utcnow().isoformat() + "Z",
            "error":            error,
        }
