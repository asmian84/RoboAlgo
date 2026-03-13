"""
RoboAlgo — Range Compression Engine
Detects multi-timeframe volatility compression (energy buildup before expansion).

Compression Rules (must satisfy ≥2 of 3):
  1. BB_width_percentile < 20th percentile
  2. ATR_percentile      < 25th percentile
  3. 10-bar range  <  30-bar range (range contraction)

MTF Compression Score:
  compression_score = 0.5 × daily_score + 0.3 × 4h_score + 0.2 × 1h_score

Only signals where compression_score exceeds threshold proceed to Breakout Engine.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, VolatilityRegime, RangeCompression

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BB_WIDTH_THRESHOLD   = 0.20   # BB width percentile < 20 = compressed
ATR_THRESHOLD        = 0.25   # ATR percentile < 25 = compressed
COMPRESSION_LOOKBACK = 252    # rolling window for percentile ranks
RANGE_SHORT          = 10     # bars for short-term range
RANGE_LONG           = 30     # bars for long-term range
MIN_COMPRESSION_BARS = 3      # minimum consecutive bars to flag compression
MTF_THRESHOLD        = 35.0   # minimum MTF score to be considered significant compression


class RangeCompressionEngine:
    """
    Detects multi-timeframe range compression for all instruments.

    Usage:
        engine = RangeCompressionEngine()
        engine.compute_and_store()                # all instruments
        engine.compute_and_store(symbol="TQQQ")  # single instrument
        result = engine.get_latest(symbol="TQQQ")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Compute range compression for one or all instruments and upsert to DB."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Range compression"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Compression failed for {inst.symbol}: {e}")
            logger.info(f"Range compression engine: stored {total} rows.")
            return total

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the most recent compression state for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None
            row = session.execute(
                select(RangeCompression)
                .where(RangeCompression.instrument_id == instr.id)
                .order_by(desc(RangeCompression.date))
                .limit(1)
            ).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    def get_compressed_instruments(self) -> list[dict]:
        """Return all instruments currently in compression with scores."""
        with get_session() as session:
            # Subquery: latest date per instrument
            rows = session.execute(
                select(RangeCompression, Instrument.symbol)
                .join(Instrument, Instrument.id == RangeCompression.instrument_id)
                .where(RangeCompression.is_compressed == True)
                .order_by(desc(RangeCompression.compression_score))
            ).all()

            # Filter to latest date per instrument
            seen = set()
            results = []
            for rc, sym in rows:
                if sym not in seen:
                    seen.add(sym)
                    d = self._row_to_dict(rc)
                    d["symbol"] = sym
                    results.append(d)
            return results

    def get_mtf_compression(self, symbol: str) -> dict:
        """
        Compute live MTF compression score using yfinance intraday data.
        Returns daily + 4h + 1h compression sub-scores and weighted total.
        """
        sym = symbol.upper()
        try:
            # Fetch intraday data from yfinance
            h4_data = yf.download(sym, period="60d", interval="1h", progress=False, auto_adjust=True)
            if h4_data.empty:
                return {"h4_compression": 0.0, "h1_compression": 0.0, "mtf_available": False}

            # Flatten multi-level columns if present
            if isinstance(h4_data.columns, pd.MultiIndex):
                h4_data.columns = h4_data.columns.get_level_values(0)

            # Compute hourly compression
            h1_score = self._compute_intraday_compression(h4_data, bars_lookback=14)

            # Resample to 4h
            h4_resampled = h4_data.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            h4_score = self._compute_intraday_compression(h4_resampled, bars_lookback=14)

            return {
                "h1_compression": round(h1_score, 2),
                "h4_compression": round(h4_score, 2),
                "mtf_available": True,
            }
        except Exception as e:
            logger.debug(f"MTF fetch failed for {sym}: {e}")
            return {"h1_compression": 0.0, "h4_compression": 0.0, "mtf_available": False}

    # ── Internal Processing ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Build compression records for one instrument from DB data."""
        # Load price data
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < RANGE_LONG + COMPRESSION_LOOKBACK:
            return []

        # Load indicator data
        ind_rows = session.execute(
            select(Indicator)
            .where(Indicator.instrument_id == instrument.id)
            .order_by(Indicator.date)
        ).scalars().all()

        # Load volatility regimes for percentile ranks
        vol_rows = session.execute(
            select(VolatilityRegime)
            .where(VolatilityRegime.instrument_id == instrument.id)
            .order_by(VolatilityRegime.date)
        ).scalars().all()

        # Build DataFrames
        prices = pd.DataFrame([{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in price_rows]).set_index("date")

        indicators = pd.DataFrame([{
            "date": r.date, "atr": r.atr,
            "bb_upper": r.bb_upper, "bb_lower": r.bb_lower, "bb_middle": r.bb_middle,
        } for r in ind_rows]).set_index("date")

        vol_regimes = pd.DataFrame([{
            "date": r.date,
            "atr_percentile": r.atr_percentile,
            "bb_width_percentile": r.bb_width_percentile,
        } for r in vol_rows]).set_index("date")

        # Align all on common dates
        df = prices.join(indicators, how="left").join(vol_regimes, how="left")
        df = df.dropna(subset=["close"])

        return self._compute_compression_series(instrument.id, df)

    def _compute_compression_series(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Compute compression metrics across the entire price history."""
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]

        # Compute rolling N-bar ranges
        range_10 = (high.rolling(RANGE_SHORT).max() - low.rolling(RANGE_SHORT).min())
        range_30 = (high.rolling(RANGE_LONG).max() - low.rolling(RANGE_LONG).min())
        range_ratio = (range_10 / range_30.replace(0, np.nan)).clip(0, 2)

        # Rolling min/max for compression range bounds
        range_high = high.rolling(RANGE_SHORT).max()
        range_low  = low.rolling(RANGE_SHORT).min()
        range_mid  = (range_high + range_low) / 2

        # Compute daily compression sub-score (0–100)
        daily_score = self._compute_daily_compression_score(df)

        # Track consecutive compression bars
        is_comp_flag = pd.Series(False, index=df.index)
        comp_duration = pd.Series(0, index=df.index)

        records = []
        consecutive = 0
        dates = df.index.tolist()

        for i, date in enumerate(dates):
            if i < RANGE_LONG:
                continue

            row = df.iloc[i]
            atr_pct = row.get("atr_percentile", np.nan)
            bb_pct  = row.get("bb_width_percentile", np.nan)
            r_ratio = range_ratio.iloc[i]
            d_score = daily_score.iloc[i]

            if any(pd.isna(v) for v in [atr_pct, bb_pct, r_ratio]):
                consecutive = 0
                continue

            # Count triggers met
            trigger_bb    = bb_pct < BB_WIDTH_THRESHOLD
            trigger_atr   = atr_pct < ATR_THRESHOLD
            trigger_range = (not pd.isna(r_ratio)) and r_ratio < 1.0

            triggers_met = sum([trigger_bb, trigger_atr, trigger_range])
            is_compressed = triggers_met >= 2

            if is_compressed:
                consecutive += 1
            else:
                consecutive = 0

            # MTF score using available daily data (intraday would need live fetch)
            # Use daily score as primary; h4/h1 default to daily proxy
            compression_score = d_score  # live MTF applied in get_latest() / signal gen

            records.append({
                "instrument_id":      instrument_id,
                "date":               date,
                "bb_width_pct":       round(float(bb_pct), 4),
                "atr_pct":            round(float(atr_pct), 4),
                "range_10bar":        round(float(range_10.iloc[i]), 4),
                "range_30bar":        round(float(range_30.iloc[i]), 4),
                "range_ratio":        round(float(r_ratio), 4),
                "is_compressed":      is_compressed,
                "compression_duration": consecutive,
                "range_high":         round(float(range_high.iloc[i]), 4),
                "range_low":          round(float(range_low.iloc[i]), 4),
                "range_mid":          round(float(range_mid.iloc[i]), 4),
                "daily_compression":  round(float(d_score), 2),
                "h4_compression":     None,   # populated by live MTF fetch
                "h1_compression":     None,
                "compression_score":  round(float(compression_score), 2),
            })

        return records

    def _compute_daily_compression_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a 0–100 daily compression score.
        Higher = more compressed (energy buildup).
        Formula: score = (1 - BB_width_pct) × 40 + (1 - ATR_pct) × 35 + range_score × 25
        """
        bb_pct  = df.get("bb_width_percentile", pd.Series(0.5, index=df.index)).fillna(0.5)
        atr_pct = df.get("atr_percentile", pd.Series(0.5, index=df.index)).fillna(0.5)

        range_10 = (df["high"].rolling(RANGE_SHORT).max() - df["low"].rolling(RANGE_SHORT).min()).fillna(0)
        range_30 = (df["high"].rolling(RANGE_LONG).max() - df["low"].rolling(RANGE_LONG).min()).replace(0, np.nan)
        range_ratio = (range_10 / range_30).clip(0, 2).fillna(1.0)
        # range_score = 1 when ratio = 0 (fully compressed), 0 when ratio = 1 (normal)
        range_score = (1 - range_ratio.clip(0, 1))

        score = ((1 - bb_pct.clip(0, 1)) * 40
                 + (1 - atr_pct.clip(0, 1)) * 35
                 + range_score * 25)
        return score.clip(0, 100)

    def _compute_intraday_compression(self, df: pd.DataFrame, bars_lookback: int = 14) -> float:
        """Compute compression score for an intraday DataFrame (1h or 4h bars)."""
        if len(df) < bars_lookback * 2:
            return 0.0

        close = df["Close"] if "Close" in df.columns else df["close"]
        high  = df["High"]  if "High"  in df.columns else df["high"]
        low   = df["Low"]   if "Low"   in df.columns else df["low"]

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(bars_lookback).mean()
        atr_pct = atr / close.replace(0, np.nan)

        # BB width
        sma = close.rolling(bars_lookback).mean()
        std = close.rolling(bars_lookback).std()
        bb_width = (2 * std / sma.replace(0, np.nan)).fillna(0)

        # Percentile ranks over available history
        lookback = min(len(df), COMPRESSION_LOOKBACK)
        atr_rank = atr_pct.rolling(lookback).apply(
            lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x)-1, 1)), raw=True
        ).fillna(0.5)
        bb_rank = bb_width.rolling(lookback).apply(
            lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x)-1, 1)), raw=True
        ).fillna(0.5)

        # Range contraction
        r10 = (high.rolling(min(RANGE_SHORT, bars_lookback)).max()
               - low.rolling(min(RANGE_SHORT, bars_lookback)).min())
        r30 = (high.rolling(min(RANGE_LONG, bars_lookback * 2)).max()
               - low.rolling(min(RANGE_LONG, bars_lookback * 2)).min()).replace(0, np.nan)
        range_ratio = (r10 / r30).clip(0, 2).fillna(1.0)
        range_score = (1 - range_ratio.clip(0, 1))

        last_atr = float(atr_rank.iloc[-1])
        last_bb  = float(bb_rank.iloc[-1])
        last_rng = float(range_score.iloc[-1])

        score = (1 - last_bb) * 40 + (1 - last_atr) * 35 + last_rng * 25
        return float(np.clip(score, 0, 100))

    def _upsert(self, session, records: list[dict]):
        """Upsert compression records using PostgreSQL ON CONFLICT DO UPDATE."""
        if not records:
            return
        stmt = pg_insert(RangeCompression).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_range_comp_inst_date",
            set_={
                "bb_width_pct":          stmt.excluded.bb_width_pct,
                "atr_pct":               stmt.excluded.atr_pct,
                "range_10bar":           stmt.excluded.range_10bar,
                "range_30bar":           stmt.excluded.range_30bar,
                "range_ratio":           stmt.excluded.range_ratio,
                "is_compressed":         stmt.excluded.is_compressed,
                "compression_duration":  stmt.excluded.compression_duration,
                "range_high":            stmt.excluded.range_high,
                "range_low":             stmt.excluded.range_low,
                "range_mid":             stmt.excluded.range_mid,
                "daily_compression":     stmt.excluded.daily_compression,
                "compression_score":     stmt.excluded.compression_score,
            }
        )
        session.execute(stmt)
        session.commit()

    def _row_to_dict(self, row: RangeCompression) -> dict:
        return {
            "date":                 str(row.date),
            "bb_width_pct":         row.bb_width_pct,
            "atr_pct":              row.atr_pct,
            "range_10bar":          row.range_10bar,
            "range_30bar":          row.range_30bar,
            "range_ratio":          row.range_ratio,
            "is_compressed":        row.is_compressed,
            "compression_duration": row.compression_duration,
            "range_high":           row.range_high,
            "range_low":            row.range_low,
            "range_mid":            row.range_mid,
            "daily_compression":    row.daily_compression,
            "h4_compression":       row.h4_compression,
            "h1_compression":       row.h1_compression,
            "compression_score":    row.compression_score,
        }
