"""
RoboAlgo — Liquidity Sweep Detection Engine
Detects failed breakouts (stop hunts) that reverse quickly after triggering
stop orders at key price levels.

A liquidity sweep occurs when price briefly violates a key level (swing high /
swing low / compression boundary), flushes trapped orders, then rapidly reverses.
Institutional players engineer these moves to fill their own orders at better
prices before the real move begins.

Score: 0–100

Components
----------
level_break  (40%)
    Checks whether the "break bar" (bar just before the reversal window) broke
    above the N-period swing high or below the N-period swing low.
    Smaller breaks score HIGHER — a 3 % break is more likely a real breakout
    than a 0.2 % sweep.
    Score = max(0, 100 − (break_pct − BREAK_FLOOR) / (BREAK_CAP − BREAK_FLOOR) × 100).
    Zero if no break detected or break < BREAK_FLOOR (0.1 %).

reversal_speed  (30%)
    After the break bar, how quickly did price close back inside the range?
    Measures bars-to-reversal over the RETEST_BARS (3-bar) window.
    Score: 100 if reversion on bar 1, 67 on bar 2, 33 on bar 3, 0 if no reversion.

wick_dominance  (30%)
    On the break bar itself, measures how large the dominant wick is relative
    to the total candle range.  A long shadow pointing in the break direction
    is the classic sweep fingerprint.
    dominant_wick = upper_wick (high sweep) | lower_wick (low sweep)
    wick_ratio = dominant_wick / candle_range
    Score = _linear_score(wick_ratio, 0.20, 0.60).

Gate
----
liquidity_sweep_score ≥ 70 → flag as reversal / trap-trade candidate.
Signal engine should surface these as counter-trend reversal signals.

Storage: liquidity_sweep_scores table.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# ── Component weights ──────────────────────────────────────────────────────────
WEIGHTS = {
    "level_break":    0.40,
    "reversal_speed": 0.30,
    "wick_dominance": 0.30,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ── Constants ──────────────────────────────────────────────────────────────────
LOOKBACK_BARS  = 30      # bars used to establish the key swing level
RETEST_BARS    = 3       # bars after the break to detect reversal
BREAK_FLOOR_PCT = 0.001  # minimum 0.1 % break to qualify as a sweep candidate
BREAK_CAP_PCT   = 0.03   # 3 % break → likely a real breakout, not a sweep
WICK_FLOOR      = 0.20   # wick ratio below this → 0 wick score
WICK_CAP        = 0.60   # wick ratio above this → 100 wick score
GATE_THRESHOLD  = 70.0   # score ≥ this → reversal/trap-trade signal


def _linear_score(value: float, floor: float, cap: float) -> float:
    """Map value linearly from 0 at floor to 100 at cap, clamped."""
    if value <= floor:
        return 0.0
    if value >= cap:
        return 100.0
    return (value - floor) / (cap - floor) * 100.0


class LiquiditySweepEngine:
    """
    Detects stop-hunt / liquidity sweep patterns in recent price action.

    A qualifying sweep requires ALL three conditions to fire:
      1. Price briefly breaks a key swing level
      2. The break bar has a dominant shadow in the break direction
      3. Price closes back inside the range within RETEST_BARS bars

    Score ≥ GATE_THRESHOLD (70) → treat as reversal / trap-trade candidate.

    Usage:
        engine = LiquiditySweepEngine()
        result = engine.detect_liquidity_sweep("SOXL")
        score  = result["liquidity_sweep_score"]   # 0–100
    """

    def detect_liquidity_sweep(
        self,
        symbol:     str,
        as_of_date: Optional[date] = None,
    ) -> dict:
        """
        Compute the Liquidity Sweep Score for *symbol*.

        Returns:
            {
              symbol, date, liquidity_sweep_score,
              sweep_type, sweep_level, break_pct,
              gate_passed, gate_threshold,
              components: {level_break, reversal_speed, wick_dominance},
              weights, computed_at
            }
        """
        target_date = as_of_date or date.today()
        needed_bars = LOOKBACK_BARS + RETEST_BARS + 5

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
                    PriceData.date, PriceData.open,
                    PriceData.high, PriceData.low,
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

        if prices.empty or len(prices) < LOOKBACK_BARS + RETEST_BARS:
            return self._empty(symbol, target_date, error="Insufficient price data")

        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values("date").reset_index(drop=True)

        # ── Identify windows ──────────────────────────────────────────────────
        # Break bar: bar at position -(RETEST_BARS + 1) from the end
        # Reversal window: the last RETEST_BARS bars
        # Historical window: everything before the break bar
        break_idx   = len(prices) - RETEST_BARS - 1
        break_bar   = prices.iloc[break_idx]
        hist        = prices.iloc[:break_idx]          # bars before break bar
        recent      = prices.iloc[break_idx + 1:]      # bars after break bar

        if hist.empty:
            return self._empty(symbol, target_date, error="Insufficient history for level detection")

        # ── Key swing levels from historical bars ─────────────────────────────
        swing_high = float(hist["high"].max())
        swing_low  = float(hist["low"].min())

        break_high  = float(break_bar["high"])
        break_low   = float(break_bar["low"])
        break_open  = float(break_bar["open"])
        break_close = float(break_bar["close"])

        # ── Component 1: Level Break ──────────────────────────────────────────
        sweep_type  = "none"
        break_pct   = 0.0
        sweep_level = 0.0

        if break_high > swing_high:
            # High sweep — price briefly broke above resistance
            sweep_level = swing_high
            break_pct   = (break_high - swing_high) / max(swing_high, 1e-6)
            sweep_type  = "high_sweep"
        elif break_low < swing_low:
            # Low sweep — price briefly broke below support
            sweep_level = swing_low
            break_pct   = (swing_low - break_low) / max(swing_low, 1e-6)
            sweep_type  = "low_sweep"

        if sweep_type == "none" or break_pct < BREAK_FLOOR_PCT:
            level_break_score = 0.0
            sweep_type        = "none"
        else:
            # Smaller breaks score HIGHER — large breaks suggest a real breakout
            normalized        = (break_pct - BREAK_FLOOR_PCT) / max(BREAK_CAP_PCT - BREAK_FLOOR_PCT, 1e-9)
            level_break_score = max(0.0, 100.0 - normalized * 100.0)

        # ── Component 2: Reversal Speed ───────────────────────────────────────
        reversal_speed_score = 0.0
        if sweep_type != "none" and not recent.empty:
            for i, (_, row) in enumerate(recent.iterrows()):
                close  = float(row["close"])
                bars_elapsed = i + 1
                reversed_in  = (
                    (sweep_type == "high_sweep" and close < sweep_level) or
                    (sweep_type == "low_sweep"  and close > sweep_level)
                )
                if reversed_in:
                    # bar 1 → 100, bar 2 → 67, bar 3 → 33
                    reversal_speed_score = max(
                        0.0,
                        100.0 * (1.0 - (bars_elapsed - 1) / RETEST_BARS),
                    )
                    break

        # ── Component 3: Wick Dominance ───────────────────────────────────────
        wick_dominance_score = 0.0
        if sweep_type != "none":
            candle_range = max(break_high - break_low, 1e-6)
            if sweep_type == "high_sweep":
                dominant_wick = break_high - max(break_open, break_close)
            else:
                dominant_wick = min(break_open, break_close) - break_low
            wick_ratio           = max(0.0, dominant_wick) / candle_range
            wick_dominance_score = _linear_score(wick_ratio, WICK_FLOOR, WICK_CAP)

        # ── Weighted total ────────────────────────────────────────────────────
        components = {
            "level_break":    round(level_break_score, 1),
            "reversal_speed": round(reversal_speed_score, 1),
            "wick_dominance": round(wick_dominance_score, 1),
        }
        sweep_score = round(
            sum(components[k] * WEIGHTS[k] for k in WEIGHTS),
            1,
        )
        gate_passed = sweep_score >= GATE_THRESHOLD

        return {
            "symbol":                symbol.upper(),
            "date":                  target_date.isoformat(),
            "liquidity_sweep_score": sweep_score,
            "gate_passed":           gate_passed,
            "gate_threshold":        GATE_THRESHOLD,
            "sweep_type":            sweep_type,
            "sweep_level":           round(sweep_level, 2) if sweep_level else None,
            "break_pct":             round(break_pct * 100, 3),  # expressed as %
            "components":            components,
            "weights":               WEIGHTS,
            "computed_at":           datetime.utcnow().isoformat() + "Z",
        }

    # ── Batch compute + store ─────────────────────────────────────────────────

    def compute_and_store_all(self, as_of_date: Optional[date] = None):
        """Compute and persist LiquiditySweepScore for every tracked instrument."""
        target_date = as_of_date or date.today()

        with get_session() as session:
            instruments = session.execute(select(Instrument)).scalars().all()

        for inst in instruments:
            try:
                result = self.detect_liquidity_sweep(inst.symbol, target_date)
                self._persist(result, inst.id, target_date)
            except Exception as e:
                logger.warning("LiquiditySweepEngine failed for %s: %s", inst.symbol, e)

    def _persist(self, result: dict, instrument_id: int, score_date: date):
        from database.models import LiquiditySweepScore

        record = {
            "instrument_id":          instrument_id,
            "symbol":                 result["symbol"],
            "date":                   score_date,
            "liquidity_sweep_score":  result["liquidity_sweep_score"],
            "sweep_type":             result.get("sweep_type", "none"),
            "sweep_level":            result.get("sweep_level"),
            "break_pct":              result.get("break_pct", 0.0),
            "level_break_score":      result["components"].get("level_break", 0.0),
            "reversal_speed_score":   result["components"].get("reversal_speed", 0.0),
            "wick_dominance_score":   result["components"].get("wick_dominance", 0.0),
        }
        update_cols = [c for c in record if c != "instrument_id"]

        with get_session() as session:
            stmt = pg_insert(LiquiditySweepScore).values([record])
            stmt = stmt.on_conflict_do_update(
                constraint="uq_liq_sweep_inst_date",
                set_={c: stmt.excluded[c] for c in update_cols},
            )
            session.execute(stmt)
            session.commit()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty(symbol: str, d: date, error: str = "") -> dict:
        return {
            "symbol":                symbol.upper(),
            "date":                  d.isoformat(),
            "liquidity_sweep_score": 0.0,
            "gate_passed":           False,
            "gate_threshold":        GATE_THRESHOLD,
            "sweep_type":            "none",
            "sweep_level":           None,
            "break_pct":             0.0,
            "components":            {},
            "weights":               WEIGHTS,
            "computed_at":           datetime.utcnow().isoformat() + "Z",
            "error":                 error,
        }
