"""
RoboAlgo — Liquidity Shelf Detection Engine
Detects price levels where liquidity is being absorbed before a breakout.
Institutional accumulation zones ("shelves") show: repeated price tests,
range compression, and rising volume while price stalls.

Score: 0–100

Components
----------
touch_count  (40%)
    Counts repeated touches of a key price level (highs or lows).
    Method: cluster extremes within a tolerance band (0.3% of price).
    Score = min(touches, 6) / 6 × 100.

range_compression (30%)
    Measures if price range is shrinking: recent_range / historical_range.
    Lower ratio → more compression → higher score.
    Score = max(0, 1 − ratio) × 100.

volume_absorption (30%)
    Slope of volume over the last N bars while price stalls.
    Positive slope (volume rising while range stalls) = absorption.
    Normalised: clip(slope_z_score, 0, 2) / 2 × 100.

Integration
-----------
SetupQualityScore: liquidity_shelf_score added at 10% weight.
Gating: if breakout detected but shelf_score < 40 → penalise setup score.

Storage: liquidity_shelf_scores table.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS = {
    "touch_count":       0.40,
    "range_compression": 0.30,
    "volume_absorption": 0.30,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

LEVEL_TOLERANCE_PCT  = 0.003   # 0.3% of price (cluster radius)
LOOKBACK_RECENT      = 10      # bars for "recent" range + volume
LOOKBACK_HISTORICAL  = 40      # bars for "historical" range baseline
MAX_TOUCHES          = 6       # touch count that gives 100 score
GATE_THRESHOLD       =  40.0   # minimum shelf score when breakout is detected


def _slope_zscore(series: pd.Series) -> float:
    """Return the z-score of the OLS slope of a series vs bar index."""
    if len(series) < 3:
        return 0.0
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    try:
        slope, _, _, _, _ = scipy_stats.linregress(x, y)
    except Exception:
        return 0.0
    std = y.std()
    if std < 1e-9:
        return 0.0
    # Express slope as fraction of std per bar, then scale to z-score
    return float(slope / std)


class LiquidityShelfEngine:
    """
    Detects institutional accumulation shelves (absorption zones).

    Usage:
        engine = LiquidityShelfEngine()
        result = engine.detect_liquidity_shelf("SOXL")
        score  = result["liquidity_shelf_score"]
    """

    def detect_liquidity_shelf(
        self,
        symbol:     str,
        as_of_date: Optional[date] = None,
    ) -> dict:
        """
        Compute the Liquidity Shelf Score for *symbol*.

        Returns:
            {
              symbol, date, liquidity_shelf_score,
              components: {touch_count, range_compression, volume_absorption},
              shelf_level, shelf_type,
              weights, gate_threshold
            }
        """
        target_date = as_of_date or date.today()
        needed_bars = LOOKBACK_HISTORICAL + 10

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

        if prices.empty or len(prices) < LOOKBACK_RECENT + 3:
            return self._empty(symbol, target_date, error="Insufficient price data")

        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values("date").reset_index(drop=True)

        latest_close = float(prices["close"].iloc[-1])
        tolerance    = latest_close * LEVEL_TOLERANCE_PCT

        # ── Component 1: Touch Count ──────────────────────────────────────────
        # Use both highs (resistance tests) and lows (support tests)
        # Find the dominant shelf level as the most-touched extreme
        all_extremes = pd.concat([prices["high"], prices["low"]])
        shelf_level, touches, shelf_type = self._find_shelf(all_extremes, tolerance)
        touch_score = min(touches / MAX_TOUCHES, 1.0) * 100.0

        # ── Component 2: Range Compression ───────────────────────────────────
        recent   = prices.iloc[-LOOKBACK_RECENT:]
        historic = prices.iloc[-LOOKBACK_HISTORICAL:-LOOKBACK_RECENT] if len(prices) >= LOOKBACK_HISTORICAL else prices.iloc[:len(prices) - LOOKBACK_RECENT]

        recent_range = float((recent["high"] - recent["low"]).mean())
        hist_range   = float((historic["high"] - historic["low"]).mean()) if not historic.empty else recent_range

        ratio = recent_range / max(hist_range, 1e-6)
        range_comp_score = max(0.0, 1.0 - ratio) * 100.0
        range_comp_score = min(range_comp_score, 100.0)

        # ── Component 3: Volume Absorption ────────────────────────────────────
        # High volume while price stalls = absorption
        recent_vol = recent["volume"].astype(float)
        vol_slope_z = _slope_zscore(recent_vol)
        vol_abs_score = min(max(vol_slope_z, 0.0), 2.0) / 2.0 * 100.0

        # ── Weighted total ────────────────────────────────────────────────────
        components = {
            "touch_count":       round(touch_score, 1),
            "range_compression": round(range_comp_score, 1),
            "volume_absorption": round(vol_abs_score, 1),
        }
        shelf_score = round(
            sum(components[k] * WEIGHTS[k] for k in WEIGHTS),
            1,
        )

        return {
            "symbol":               symbol.upper(),
            "date":                 target_date.isoformat(),
            "liquidity_shelf_score": shelf_score,
            "shelf_level":          round(shelf_level, 2) if shelf_level else None,
            "shelf_type":           shelf_type,
            "touch_count":          touches,
            "gate_threshold":       GATE_THRESHOLD,
            "components":           components,
            "weights":              WEIGHTS,
            "computed_at":          datetime.utcnow().isoformat() + "Z",
        }

    # ── Batch compute + store ─────────────────────────────────────────────────

    def compute_and_store_all(self, as_of_date: Optional[date] = None):
        """Compute and persist for every tracked instrument."""
        from database.models import LiquidityShelfScore

        target_date = as_of_date or date.today()
        with get_session() as session:
            instruments = session.execute(select(Instrument)).scalars().all()

        for inst in instruments:
            try:
                result = self.detect_liquidity_shelf(inst.symbol, target_date)
                self._persist(result, inst.id, target_date)
            except Exception as e:
                logger.warning("LiquidityShelfEngine failed for %s: %s", inst.symbol, e)

    def _persist(self, result: dict, instrument_id: int, score_date: date):
        from database.models import LiquidityShelfScore

        record = {
            "instrument_id":        instrument_id,
            "symbol":               result["symbol"],
            "date":                 score_date,
            "liquidity_shelf_score": result["liquidity_shelf_score"],
            "shelf_level":          result.get("shelf_level"),
            "shelf_type":           result.get("shelf_type"),
            "touch_count":          result.get("touch_count", 0),
            "touch_count_score":    result["components"].get("touch_count", 0),
            "range_compression_score": result["components"].get("range_compression", 0),
            "volume_absorption_score": result["components"].get("volume_absorption", 0),
        }
        update_cols = [c for c in record if c != "instrument_id"]

        with get_session() as session:
            stmt = pg_insert(LiquidityShelfScore).values([record])
            stmt = stmt.on_conflict_do_update(
                constraint="uq_liq_shelf_inst_date",
                set_={c: stmt.excluded[c] for c in update_cols},
            )
            session.execute(stmt)
            session.commit()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_shelf(
        extremes: pd.Series,
        tolerance: float,
    ) -> tuple[float, int, str]:
        """
        Cluster price extremes to find the most-tested level.
        Returns (level, touch_count, "support"|"resistance"|"unknown").
        """
        if extremes.empty:
            return 0.0, 0, "unknown"

        vals = extremes.values
        best_level  = float(vals[0])
        best_touches = 0

        for anchor in vals:
            touches = int(np.sum(np.abs(vals - anchor) <= tolerance))
            if touches > best_touches:
                best_touches = touches
                best_level   = float(anchor)

        # Determine if it's support (below current price) or resistance
        return best_level, best_touches, "unknown"

    @staticmethod
    def _empty(symbol: str, d: date, error: str = "") -> dict:
        return {
            "symbol":                symbol.upper(),
            "date":                  d.isoformat(),
            "liquidity_shelf_score": 0.0,
            "shelf_level":           None,
            "shelf_type":            "unknown",
            "touch_count":           0,
            "gate_threshold":        GATE_THRESHOLD,
            "components":            {},
            "weights":               WEIGHTS,
            "computed_at":           datetime.utcnow().isoformat() + "Z",
            "error":                 error,
        }
