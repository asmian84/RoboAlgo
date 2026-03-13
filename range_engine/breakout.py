"""
RoboAlgo — Breakout Engine
Detects breakouts from range compression with multi-trigger validation.

Breakout Conditions (require ≥2 of 3):
  1. Price trigger:    close > compression_range_high (or < range_low for shorts)
  2. Volume trigger:   volume_ratio > 1.5× 20-day average
  3. Momentum trigger: RSI rising ≥3 bars + MACD histogram positive

Breakout Strength Formula:
  breakout_strength = 0.35 × compression_duration_score
                    + 0.30 × volume_ratio_score
                    + 0.20 × breakout_distance_score
                    + 0.15 × volatility_expansion_speed_score
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import (
    Instrument, PriceData, Indicator, RangeCompression, BreakoutSignal
)

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
VOLUME_RATIO_MIN     = 1.5    # volume must be 1.5× average to trigger
MOMENTUM_RSI_BARS    = 3      # RSI must rise for N consecutive bars
ATR_LOOKBACK         = 14     # ATR period
VOLUME_AVG_LOOKBACK  = 20     # volume moving average period
MIN_COMP_DURATION    = 3      # minimum compression bars required before valid breakout
MIN_TRIGGERS         = 2      # minimum triggers needed to classify as breakout


class BreakoutEngine:
    """
    Detects breakout events from range compressions.

    Usage:
        engine = BreakoutEngine()
        engine.compute_and_store()               # all instruments
        engine.compute_and_store(symbol="SOXL")
        latest = engine.get_latest_breakout("SOXL")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Compute breakout signals for one or all instruments."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Breakout detection"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Breakout detection failed for {inst.symbol}: {e}")
            logger.info(f"Breakout engine: stored {total} breakout signals.")
            return total

    def get_latest_breakout(self, symbol: str) -> Optional[dict]:
        """Return the most recent breakout signal for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None
            row = session.execute(
                select(BreakoutSignal)
                .where(BreakoutSignal.instrument_id == instr.id)
                .order_by(desc(BreakoutSignal.date))
                .limit(1)
            ).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    def get_active_breakouts(self, min_strength: float = 50.0) -> list[dict]:
        """Return all recent breakout signals above a strength threshold."""
        with get_session() as session:
            rows = session.execute(
                select(BreakoutSignal, Instrument.symbol)
                .join(Instrument, Instrument.id == BreakoutSignal.instrument_id)
                .where(BreakoutSignal.breakout_strength >= min_strength)
                .where(BreakoutSignal.triggers_met >= MIN_TRIGGERS)
                .order_by(desc(BreakoutSignal.date), desc(BreakoutSignal.breakout_strength))
            ).all()

            seen = set()
            results = []
            for bs, sym in rows:
                if sym not in seen:
                    seen.add(sym)
                    d = self._row_to_dict(bs)
                    d["symbol"] = sym
                    results.append(d)
            return results

    # ── Internal Processing ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Compute breakout records for one instrument."""
        # Load price data
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < VOLUME_AVG_LOOKBACK + 10:
            return []

        # Load indicators
        ind_rows = session.execute(
            select(Indicator)
            .where(Indicator.instrument_id == instrument.id)
            .order_by(Indicator.date)
        ).scalars().all()

        # Load compression data
        comp_rows = session.execute(
            select(RangeCompression)
            .where(RangeCompression.instrument_id == instrument.id)
            .order_by(RangeCompression.date)
        ).scalars().all()

        if not comp_rows:
            return []

        # Build DataFrames
        prices = pd.DataFrame([{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in price_rows]).set_index("date")

        indicators = pd.DataFrame([{
            "date": r.date, "rsi": r.rsi, "atr": r.atr,
            "macd_line": r.macd_line, "macd_signal": r.macd_signal,
            "macd_histogram": r.macd_histogram,
        } for r in ind_rows]).set_index("date")

        compressions = pd.DataFrame([{
            "date":                 r.date,
            "is_compressed":        r.is_compressed,
            "compression_duration": r.compression_duration,
            "compression_score":    r.compression_score,
            "range_high":           r.range_high,
            "range_low":            r.range_low,
        } for r in comp_rows]).set_index("date")

        df = prices.join(indicators, how="left").join(compressions, how="left")
        df = df.dropna(subset=["close"])

        return self._detect_breakouts(instrument.id, df)

    def _detect_breakouts(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Detect breakout events across full price history."""
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"].fillna(0)
        rsi    = df.get("rsi", pd.Series(50.0, index=df.index)).fillna(50)
        atr    = df.get("atr", pd.Series(0.0, index=df.index)).fillna(0)
        macd_h = df.get("macd_histogram", pd.Series(0.0, index=df.index)).fillna(0)

        # Compute volume ratio
        vol_avg   = volume.rolling(VOLUME_AVG_LOOKBACK).mean()
        vol_ratio = (volume / vol_avg.replace(0, np.nan)).fillna(1.0)

        # ATR expansion speed (rate of change of ATR)
        atr_expansion = (atr / atr.shift(5).replace(0, np.nan) - 1).fillna(0).clip(-1, 5)

        records = []
        dates = df.index.tolist()

        # Track prior compression state
        was_compressed = False
        prior_range_high = None
        prior_range_low  = None
        prior_duration   = 0
        prior_comp_score = 0.0

        for i, date in enumerate(dates):
            if i < VOLUME_AVG_LOOKBACK:
                continue

            row           = df.iloc[i]
            cur_close     = float(close.iloc[i])
            cur_vol_ratio = float(vol_ratio.iloc[i])
            cur_atr       = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
            cur_atr_exp   = float(atr_expansion.iloc[i])
            is_comp       = bool(row.get("is_compressed", False))
            r_high        = row.get("range_high")
            r_low         = row.get("range_low")
            comp_duration = int(row.get("compression_duration", 0))
            comp_score    = float(row.get("compression_score", 0) or 0)

            # Carry forward last known range when exiting compression
            if is_comp and not pd.isna(r_high) and not pd.isna(r_low):
                prior_range_high = float(r_high)
                prior_range_low  = float(r_low)
                prior_duration   = comp_duration
                prior_comp_score = comp_score
                was_compressed = True

            # Check for breakout: was compressed, now breaking out
            if not is_comp and was_compressed and prior_range_high is not None:
                if prior_duration < MIN_COMP_DURATION:
                    was_compressed = False
                    continue

                # Evaluate breakout direction
                direction    = None
                price_trigger = False

                if cur_close > prior_range_high:
                    direction    = "up"
                    price_trigger = True
                elif cur_close < prior_range_low:
                    direction    = "down"
                    price_trigger = True

                if direction is None:
                    continue  # price still inside range, not yet a breakout

                # Volume trigger
                volume_trigger = cur_vol_ratio >= VOLUME_RATIO_MIN

                # Momentum trigger: RSI rising + MACD positive
                rsi_rising = all(
                    rsi.iloc[max(0, i-j)] < rsi.iloc[max(0, i-j+1)]
                    for j in range(MOMENTUM_RSI_BARS, 0, -1)
                ) if i >= MOMENTUM_RSI_BARS else False
                macd_positive  = float(macd_h.iloc[i]) > 0
                momentum_trigger = rsi_rising and macd_positive

                triggers_met = sum([price_trigger, volume_trigger, momentum_trigger])

                if triggers_met < MIN_TRIGGERS:
                    # Soft breakout - still store with low strength
                    pass

                # Compute breakout strength components (each 0–100)
                comp_dur_score  = min(prior_duration / 20 * 100, 100)  # max at 20 bars
                vol_score       = min((cur_vol_ratio - 1) / 3 * 100, 100)  # max at 4×
                dist_score = 0.0
                if cur_atr > 0:
                    dist = abs(cur_close - (prior_range_high if direction == "up" else prior_range_low))
                    dist_score = min(dist / cur_atr * 50, 100)  # 2 ATR = 100
                exp_score = min(max(cur_atr_exp, 0) / 0.5 * 100, 100)  # 50% ATR expansion = 100

                breakout_strength = (
                    0.35 * comp_dur_score
                    + 0.30 * vol_score
                    + 0.20 * dist_score
                    + 0.15 * exp_score
                )
                breakout_strength = round(float(np.clip(breakout_strength, 0, 100)), 2)

                # Momentum score (0–100)
                momentum_score = (
                    (float(rsi.iloc[i]) - 50) / 50 * 50 +   # RSI contribution
                    (50.0 if macd_positive else 0.0)          # MACD contribution
                )
                momentum_score = round(float(np.clip(momentum_score, 0, 100)), 2)

                records.append({
                    "instrument_id":             instrument_id,
                    "date":                      date,
                    "breakout_direction":        direction,
                    "breakout_price":            round(cur_close, 4),
                    "compression_range_high":    round(prior_range_high, 4),
                    "compression_range_low":     round(prior_range_low, 4),
                    "price_trigger":             price_trigger,
                    "volume_trigger":            volume_trigger,
                    "momentum_trigger":          momentum_trigger,
                    "triggers_met":              triggers_met,
                    "breakout_distance":         round(float(np.abs(
                        cur_close - (prior_range_high if direction == "up" else prior_range_low)
                    )), 4),
                    "volume_ratio":              round(cur_vol_ratio, 4),
                    "momentum_score":            momentum_score,
                    "volatility_expansion_speed": round(max(cur_atr_exp, 0), 4),
                    "compression_duration":      prior_duration,
                    "compression_score":         round(prior_comp_score, 2),
                    "breakout_strength":         breakout_strength,
                })

                was_compressed = False  # reset after breakout recorded

        return records

    def compute_breakout_strength(
        self,
        compression_duration: int,
        volume_ratio: float,
        breakout_distance: float,
        atr: float,
        volatility_expansion_speed: float,
    ) -> float:
        """
        Standalone breakout strength calculator for live use.
        Returns 0–100 score.
        """
        comp_dur_score = min(compression_duration / 20 * 100, 100)
        vol_score      = min((max(volume_ratio, 1) - 1) / 3 * 100, 100)
        dist_score     = min(breakout_distance / max(atr, 1e-8) * 50, 100) if atr else 0
        exp_score      = min(max(volatility_expansion_speed, 0) / 0.5 * 100, 100)

        strength = (
            0.35 * comp_dur_score
            + 0.30 * vol_score
            + 0.20 * dist_score
            + 0.15 * exp_score
        )
        return round(float(np.clip(strength, 0, 100)), 2)

    def _upsert(self, session, records: list[dict]):
        if not records:
            return
        stmt = pg_insert(BreakoutSignal).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_breakout_inst_date",
            set_={
                "breakout_direction":         stmt.excluded.breakout_direction,
                "breakout_price":             stmt.excluded.breakout_price,
                "compression_range_high":     stmt.excluded.compression_range_high,
                "compression_range_low":      stmt.excluded.compression_range_low,
                "price_trigger":              stmt.excluded.price_trigger,
                "volume_trigger":             stmt.excluded.volume_trigger,
                "momentum_trigger":           stmt.excluded.momentum_trigger,
                "triggers_met":               stmt.excluded.triggers_met,
                "breakout_distance":          stmt.excluded.breakout_distance,
                "volume_ratio":               stmt.excluded.volume_ratio,
                "momentum_score":             stmt.excluded.momentum_score,
                "volatility_expansion_speed": stmt.excluded.volatility_expansion_speed,
                "compression_duration":       stmt.excluded.compression_duration,
                "compression_score":          stmt.excluded.compression_score,
                "breakout_strength":          stmt.excluded.breakout_strength,
            }
        )
        session.execute(stmt)
        session.commit()

    def _row_to_dict(self, row: BreakoutSignal) -> dict:
        return {
            "date":                      str(row.date),
            "breakout_direction":        row.breakout_direction,
            "breakout_price":            row.breakout_price,
            "compression_range_high":    row.compression_range_high,
            "compression_range_low":     row.compression_range_low,
            "price_trigger":             row.price_trigger,
            "volume_trigger":            row.volume_trigger,
            "momentum_trigger":          row.momentum_trigger,
            "triggers_met":              row.triggers_met,
            "breakout_distance":         row.breakout_distance,
            "volume_ratio":              row.volume_ratio,
            "momentum_score":            row.momentum_score,
            "volatility_expansion_speed": row.volatility_expansion_speed,
            "compression_duration":      row.compression_duration,
            "compression_score":         row.compression_score,
            "breakout_strength":         row.breakout_strength,
        }
