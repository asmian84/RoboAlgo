"""
RoboAlgo — Breakout Quality Engine
Filters weak breakouts before they generate trade signals.

A breakout is considered "real" when volume, momentum, and candle structure
all confirm the move. Fake breakouts lack at least two of these.

Score: 0–100

Components
----------
volume_confirmation  (30%)
    volume_participation feature ≥ 1.5.
    Score scales linearly from 0 at vol_part=0.5 → 100 at vol_part=3.0.

momentum_continuation (30%)
    momentum_acceleration > 0 → positive contribution.
    Normalized: max(0, momentum_acceleration) / 1.0 → 0-1.

candle_quality (20%)
    body_ratio = |close − open| / (high − low + ε)
    Full score when body_ratio ≥ 0.6; zero when ≤ 0.2.
    Linearly interpolated between 0.2 and 0.6.

retest_stability (20%)
    Detects whether price quickly returned inside the compression range
    after the breakout candle.
    If a fast retest occurs → penalise (reduce score toward 0).
    Measured as: did close drop back below breakout level within 3 bars?
    Stable = 100, retest detected = 20.

Integration
-----------
SetupQualityScore gate:
    If breakout_quality_score < 60 → reject trade signal
    (enforced in analytics_engine/setup_quality.py)

Storage: breakout_quality_scores table (upsert by symbol + date).
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PriceData, Feature

logger = logging.getLogger(__name__)

# ── Component weights ─────────────────────────────────────────────────────────
WEIGHTS = {
    "volume_confirmation":  0.30,
    "momentum_continuation": 0.30,
    "candle_quality":       0.20,
    "retest_stability":     0.20,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ── Constants ─────────────────────────────────────────────────────────────────
VOL_PART_FLOOR = 0.5    # vol_participation below this → 0 score
VOL_PART_CAP   = 3.0    # vol_participation above this → 100 score
CANDLE_FLOOR   = 0.20   # body_ratio below this → 0 candle score
CANDLE_CAP     = 0.60   # body_ratio above this → 100 candle score
RETEST_BARS    = 3       # look-back bars to detect fast retest
STABLE_SCORE   = 100.0
RETEST_SCORE   =  20.0
GATE_THRESHOLD =  60.0   # minimum score to allow a breakout signal


def _linear_score(value: float, floor: float, cap: float) -> float:
    """Map value linearly from 0 at floor to 100 at cap, clamped."""
    if value <= floor:
        return 0.0
    if value >= cap:
        return 100.0
    return (value - floor) / (cap - floor) * 100.0


class BreakoutQualityEngine:
    """
    Computes a Breakout Quality Score for the latest price data of a symbol.

    Usage:
        engine = BreakoutQualityEngine()
        result = engine.calculate_breakout_quality("SOXL")
        score  = result["breakout_quality_score"]   # 0–100
    """

    def calculate_breakout_quality(
        self,
        symbol:     str,
        as_of_date: Optional[date] = None,
        lookback:   int = 20,
    ) -> dict:
        """
        Compute the Breakout Quality Score for *symbol*.

        Args:
            symbol:     Ticker (case-insensitive).
            as_of_date: Evaluate as of this date (default: today).
            lookback:   Number of recent bars to analyse.

        Returns:
            {
              symbol, date, breakout_quality_score,
              components: {volume_confirmation, momentum_continuation,
                           candle_quality, retest_stability},
              weights, gate_passed
            }
        """
        target_date = as_of_date or date.today()

        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(
                    Instrument.symbol == symbol.upper()
                )
            ).scalar_one_or_none()
            if inst is None:
                return self._empty(symbol, target_date, error=f"Symbol {symbol} not found")

            iid = inst.id

            # ── Raw price data (OHLCV) ────────────────────────────────────────
            prices = pd.read_sql(
                select(
                    PriceData.date, PriceData.open,
                    PriceData.high, PriceData.low,
                    PriceData.close, PriceData.volume,
                )
                .where(
                    PriceData.instrument_id == iid,
                    PriceData.date <= target_date,
                )
                .order_by(desc(PriceData.date))
                .limit(lookback + RETEST_BARS + 5),
                session.bind,
            )

            # ── v2 feature row (momentum_acceleration, volume_participation) ──
            feat_row = session.execute(
                select(
                    Feature.momentum_acceleration,
                    Feature.volume_participation,
                    Feature.volume_ratio,
                )
                .where(
                    Feature.instrument_id == iid,
                    Feature.date <= target_date,
                )
                .order_by(desc(Feature.date))
                .limit(1)
            ).one_or_none()

        if prices.empty:
            return self._empty(symbol, target_date, error="No price data")

        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values("date").reset_index(drop=True)

        latest = prices.iloc[-1]

        # ── Component 1: Volume Confirmation ─────────────────────────────────
        # volume_participation from Feature (-1→+1); we want the positive side
        # also use volume_ratio as a fallback
        if feat_row and feat_row.volume_participation is not None:
            vp = float(feat_row.volume_participation)
            # Re-map: feature is -1→+1; vol_confirmation cares about |vp| scaled
            # Treat 1.0 as the "1.5× participation" equivalent
            effective_vol = max(0.0, (vp + 1.0))  # 0→2 range
            vol_conf_score = _linear_score(effective_vol, 1.0, 3.0)
        elif feat_row and feat_row.volume_ratio is not None:
            vol_conf_score = _linear_score(float(feat_row.volume_ratio), VOL_PART_FLOOR, VOL_PART_CAP)
        else:
            # Fallback: compute volume ratio from raw prices
            avg_vol = prices["volume"].iloc[:-1].mean()
            current_vol = float(latest["volume"])
            raw_ratio = current_vol / max(avg_vol, 1)
            vol_conf_score = _linear_score(raw_ratio, VOL_PART_FLOOR, VOL_PART_CAP)

        # ── Component 2: Momentum Continuation ───────────────────────────────
        if feat_row and feat_row.momentum_acceleration is not None:
            ma = float(feat_row.momentum_acceleration)  # -1→+1
            mom_score = max(0.0, ma) * 100.0  # only positive acceleration counts
        else:
            # Fallback: 3-bar return direction
            if len(prices) >= 4:
                ret3 = (prices["close"].iloc[-1] - prices["close"].iloc[-4]) / prices["close"].iloc[-4]
                mom_score = max(0.0, min(ret3 * 2000, 100.0))
            else:
                mom_score = 50.0

        # ── Component 3: Candle Quality ───────────────────────────────────────
        hi, lo, op, cl = (
            float(latest["high"]), float(latest["low"]),
            float(latest["open"]), float(latest["close"]),
        )
        hl_range   = max(hi - lo, 1e-6)
        body_ratio = abs(cl - op) / hl_range
        candle_score = _linear_score(body_ratio, CANDLE_FLOOR, CANDLE_CAP)

        # ── Component 4: Retest Stability ─────────────────────────────────────
        # Determine breakout level = high of the bar before the latest bar
        retest_score = STABLE_SCORE  # assume stable unless we detect a retest
        if len(prices) >= RETEST_BARS + 1:
            breakout_level = float(prices["close"].iloc[-(RETEST_BARS + 1)])
            recent_closes  = prices["close"].iloc[-RETEST_BARS:].values
            if any(c < breakout_level for c in recent_closes):
                retest_score = RETEST_SCORE  # price dipped back → penalise

        # ── Weighted total ────────────────────────────────────────────────────
        components = {
            "volume_confirmation":   round(vol_conf_score, 1),
            "momentum_continuation": round(mom_score, 1),
            "candle_quality":        round(candle_score, 1),
            "retest_stability":      round(retest_score, 1),
        }

        quality_score = round(
            sum(components[k] * WEIGHTS[k] for k in WEIGHTS),
            1,
        )
        gate_passed = quality_score >= GATE_THRESHOLD

        return {
            "symbol":                 symbol.upper(),
            "date":                   target_date.isoformat(),
            "breakout_quality_score": quality_score,
            "gate_passed":            gate_passed,
            "gate_threshold":         GATE_THRESHOLD,
            "components":             components,
            "weights":                WEIGHTS,
            "computed_at":            datetime.utcnow().isoformat() + "Z",
        }

    # ── Batch compute and store ───────────────────────────────────────────────

    def compute_and_store_all(self, as_of_date: Optional[date] = None):
        """Compute and persist BreakoutQualityScore for every tracked instrument."""
        from database.models import BreakoutQualityScore

        target_date = as_of_date or date.today()

        with get_session() as session:
            instruments = session.execute(
                select(Instrument)
            ).scalars().all()

        for inst in instruments:
            try:
                result = self.calculate_breakout_quality(inst.symbol, target_date)
                self._persist(result, inst.id, target_date)
            except Exception as e:
                logger.warning("BreakoutQualityEngine failed for %s: %s", inst.symbol, e)

    def _persist(self, result: dict, instrument_id: int, score_date: date):
        from database.models import BreakoutQualityScore

        record = {
            "instrument_id":          instrument_id,
            "symbol":                 result["symbol"],
            "date":                   score_date,
            "breakout_quality_score": result["breakout_quality_score"],
            "volume_confirmation":    result["components"]["volume_confirmation"],
            "momentum_continuation":  result["components"]["momentum_continuation"],
            "candle_quality":         result["components"]["candle_quality"],
            "retest_stability":       result["components"]["retest_stability"],
        }
        update_cols = [c for c in record if c not in ("instrument_id",)]

        with get_session() as session:
            stmt = pg_insert(BreakoutQualityScore).values([record])
            stmt = stmt.on_conflict_do_update(
                constraint="uq_breakout_quality_inst_date",
                set_={c: stmt.excluded[c] for c in update_cols},
            )
            session.execute(stmt)
            session.commit()

    @staticmethod
    def _empty(symbol: str, d: date, error: str = "") -> dict:
        return {
            "symbol":                 symbol.upper(),
            "date":                   d.isoformat(),
            "breakout_quality_score": 0.0,
            "gate_passed":            False,
            "gate_threshold":         GATE_THRESHOLD,
            "components":             {},
            "weights":                WEIGHTS,
            "computed_at":            datetime.utcnow().isoformat() + "Z",
            "error":                  error,
        }
