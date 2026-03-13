"""
RoboAlgo — Volatility Regime Engine
Classifies market volatility into LOW_VOL / NORMAL_VOL / HIGH_VOL.
Detects compression (energy buildup) and expansion (breakout events).

Spec rules:
  LOW_VOL    → no trading allowed
  NORMAL_VOL → limited trading
  HIGH_VOL   → active trading (full signal generation)

Compression:
  BB_width_percentile < 15% AND ATR_percentile < 20%
  → Energy buildup detected

Expansion:
  Price breaks compression range + volume_ratio > 1.5 + momentum increasing
  → Explosive move beginning
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from config.settings import VOLATILITY_PARAMS, VOL_LOW, VOL_NORMAL, VOL_HIGH
from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, Feature, VolatilityRegime

logger = logging.getLogger(__name__)

_P = VOLATILITY_PARAMS


class VolatilityRegimeEngine:
    """
    Computes and stores volatility regimes for all instruments.

    Usage:
        engine = VolatilityRegimeEngine()
        engine.compute_and_store()                # all instruments
        engine.compute_and_store(symbol="TQQQ")  # single instrument
        regime = engine.get_latest_regime("TQQQ")
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None):
        """Compute volatility regimes for one or all instruments and upsert to DB."""
        session = get_session()
        try:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Volatility regimes"):
                rows = self._process_instrument(session, inst)
                if rows:
                    self._upsert(session, rows)
                    total += len(rows)

            logger.info(f"Volatility regime engine: stored {total} rows.")
            return total
        finally:
            session.close()

    def get_latest_regime(self, symbol: str) -> Optional[dict]:
        """Return the most recent volatility regime for a symbol."""
        session = get_session()
        try:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None

            row = session.execute(
                select(VolatilityRegime)
                .where(VolatilityRegime.instrument_id == instr.id)
                .order_by(desc(VolatilityRegime.date))
                .limit(1)
            ).scalar_one_or_none()

            return self._row_to_dict(row) if row else None
        finally:
            session.close()

    def get_history(self, symbol: str, limit: int = 500) -> list[dict]:
        """Return volatility regime history for a symbol."""
        session = get_session()
        try:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return []

            rows = session.execute(
                select(VolatilityRegime)
                .where(VolatilityRegime.instrument_id == instr.id)
                .order_by(desc(VolatilityRegime.date))
                .limit(limit)
            ).scalars().all()

            return [self._row_to_dict(r) for r in reversed(rows)]
        finally:
            session.close()

    def get_regime_summary(self) -> dict:
        """Return latest regime for all instruments grouped by regime label."""
        session = get_session()
        try:
            instruments = session.execute(
                select(Instrument).order_by(Instrument.symbol)
            ).scalars().all()

            result = {VOL_LOW: [], VOL_NORMAL: [], VOL_HIGH: []}
            for inst in instruments:
                row = session.execute(
                    select(VolatilityRegime)
                    .where(VolatilityRegime.instrument_id == inst.id)
                    .order_by(desc(VolatilityRegime.date))
                    .limit(1)
                ).scalar_one_or_none()

                if row:
                    regime = row.regime or VOL_NORMAL
                    if regime in result:
                        result[regime].append({
                            "symbol":         inst.symbol,
                            "regime":         regime,
                            "is_compression": row.is_compression,
                            "is_expansion":   row.is_expansion,
                            "atr_percentile": round(row.atr_percentile or 0, 3),
                            "bb_pct":         round(row.bb_width_percentile or 0, 3),
                            "date":           str(row.date),
                        })
            return result
        finally:
            session.close()

    # ── Core Computation ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Load price + indicator data and compute full volatility regime history."""
        # ── Load prices ────────────────────────────────────────────────────
        price_rows = pd.read_sql(
            select(
                PriceData.date,
                PriceData.open,
                PriceData.high,
                PriceData.low,
                PriceData.close,
                PriceData.volume,
            ).where(PriceData.instrument_id == instrument.id)
             .order_by(PriceData.date),
            session.bind,
        )
        if len(price_rows) < 30:
            return []

        price_rows["date"] = pd.to_datetime(price_rows["date"])
        price_rows = price_rows.set_index("date")

        # ── Load indicators ────────────────────────────────────────────────
        ind_rows = pd.read_sql(
            select(
                Indicator.date,
                Indicator.atr,
                Indicator.bb_upper,
                Indicator.bb_lower,
                Indicator.bb_middle,
                Indicator.bb_width,
                Indicator.rsi,
            ).where(Indicator.instrument_id == instrument.id)
             .order_by(Indicator.date),
            session.bind,
        )
        if ind_rows.empty:
            return []
        ind_rows["date"] = pd.to_datetime(ind_rows["date"])
        ind_rows = ind_rows.set_index("date")

        # ── Merge ──────────────────────────────────────────────────────────
        df = price_rows.join(ind_rows, how="inner")
        if len(df) < 30:
            return []

        # ── Compute metrics ────────────────────────────────────────────────
        df["atr_pct"] = self._compute_atr_pct(df["atr"], df["close"])
        df["bb_width_calc"] = self._compute_bb_width(
            df.get("bb_upper"), df.get("bb_lower"), df.get("bb_middle"),
            df.get("bb_width"),
        )
        df["realized_vol"] = self._compute_realized_vol(df["close"])
        df["volume_ratio"] = self._compute_volume_ratio(df["volume"])

        # ── Percentile ranks ───────────────────────────────────────────────
        lookback = _P["percentile_lookback"]
        df["atr_pct_rank"]  = self._rolling_percentile(df["atr_pct"],      lookback)
        df["bb_width_rank"] = self._rolling_percentile(df["bb_width_calc"], lookback)
        df["rvol_rank"]     = self._rolling_percentile(df["realized_vol"],  lookback)

        # ── Composite rank → regime ────────────────────────────────────────
        df["composite_rank"] = (df["atr_pct_rank"] + df["bb_width_rank"] + df["rvol_rank"]) / 3.0
        df["regime"] = df["composite_rank"].map(self._classify_regime)

        # ── Compression flag ───────────────────────────────────────────────
        df["is_compression"] = self._detect_compression(df["bb_width_rank"], df["atr_pct_rank"])

        # ── Expansion flag ─────────────────────────────────────────────────
        expansion_results = self._detect_expansion(
            df["close"], df["high"], df["low"],
            df["volume_ratio"], df["rsi"],
            df["is_compression"],
        )
        df["is_expansion"]          = expansion_results["is_expansion"]
        df["compression_range_high"] = expansion_results["range_high"]
        df["compression_range_low"]  = expansion_results["range_low"]

        # ── Build records ──────────────────────────────────────────────────
        records = []
        for dt, row in df.iterrows():
            if pd.isna(row.get("atr_pct")) or pd.isna(row.get("realized_vol")):
                continue
            records.append({
                "instrument_id":         instrument.id,
                "date":                  dt.date(),
                "atr_pct":               self._safe(row.get("atr_pct")),
                "bb_width":              self._safe(row.get("bb_width_calc")),
                "realized_vol_20d":      self._safe(row.get("realized_vol")),
                "atr_percentile":        self._safe(row.get("atr_pct_rank")),
                "bb_width_percentile":   self._safe(row.get("bb_width_rank")),
                "vol_percentile":        self._safe(row.get("rvol_rank")),
                "regime":                row.get("regime", VOL_NORMAL),
                "is_compression":        bool(row.get("is_compression", False)),
                "is_expansion":          bool(row.get("is_expansion", False)),
                "compression_range_high": self._safe(row.get("compression_range_high")),
                "compression_range_low":  self._safe(row.get("compression_range_low")),
            })

        return records

    # ── Metric Computation ──────────────────────────────────────────────────

    def _compute_atr_pct(self, atr: pd.Series, close: pd.Series) -> pd.Series:
        """ATR as fraction of price (normalized across instruments)."""
        return (atr / close.replace(0, np.nan)).round(6)

    def _compute_bb_width(
        self,
        bb_upper,
        bb_lower,
        bb_mid,
        bb_width_stored,
    ) -> pd.Series:
        """
        BB width = (upper - lower) / middle.
        Use stored bb_width if available (already computed by indicator engine),
        otherwise recompute from upper/lower/middle.
        """
        if bb_width_stored is not None and not bb_width_stored.isna().all():
            return bb_width_stored.ffill()
        if bb_upper is not None and bb_lower is not None and bb_mid is not None:
            return ((bb_upper - bb_lower) / bb_mid.replace(0, np.nan)).round(6)
        return pd.Series(np.nan, index=(bb_upper.index if bb_upper is not None else []))

    def _compute_realized_vol(self, close: pd.Series, window: int = None) -> pd.Series:
        """20-day annualized realized volatility from log returns."""
        w = window or _P["realized_vol_window"]
        log_ret = np.log(close / close.shift(1))
        return (log_ret.rolling(w, min_periods=w // 2).std() * np.sqrt(252)).round(6)

    def _compute_volume_ratio(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume relative to its 20-day rolling average."""
        vol_ma = volume.rolling(window, min_periods=5).mean()
        return (volume / vol_ma.replace(0, np.nan)).round(4)

    def _rolling_percentile(self, series: pd.Series, lookback: int) -> pd.Series:
        """
        Rolling percentile rank of each value within its lookback window.
        Returns 0.0–1.0.  min_periods = lookback // 2 so early bars get partial rank.
        """
        def _rank(arr):
            if len(arr) == 0:
                return np.nan
            val = arr[-1]
            if np.isnan(val):
                return np.nan
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                return np.nan
            return float((valid < val).sum()) / len(valid)

        return series.rolling(lookback, min_periods=lookback // 2).apply(
            _rank, raw=True
        ).round(4)

    def _classify_regime(self, composite_rank: float) -> str:
        """Map composite percentile rank to regime label."""
        if np.isnan(composite_rank):
            return VOL_NORMAL
        if composite_rank < _P["low_vol_composite_pct"]:
            return VOL_LOW
        if composite_rank > _P["high_vol_composite_pct"]:
            return VOL_HIGH
        return VOL_NORMAL

    def _detect_compression(
        self,
        bb_width_pct: pd.Series,
        atr_pct: pd.Series,
    ) -> pd.Series:
        """
        Compression = BB_width_percentile < 15% AND ATR_percentile < 20%.
        Both conditions must be true simultaneously.
        """
        bb_ok  = bb_width_pct < _P["compression_bb_pct"]
        atr_ok = atr_pct < _P["compression_atr_pct"]
        return (bb_ok & atr_ok).fillna(False)

    def _detect_expansion(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume_ratio: pd.Series,
        rsi: pd.Series,
        is_compression: pd.Series,
    ) -> pd.DataFrame:
        """
        Expansion = price breaks compression range
                    AND volume_ratio > 1.5
                    AND momentum (RSI) rising over 3 bars.

        Compression range = max(high) / min(low) over prior N bars
        while is_compression was True.
        """
        n = _P["expansion_range_bars"]
        mom_bars = _P["expansion_momentum_bars"]

        # Rolling compression window high/low (shift by 1 to avoid look-ahead)
        range_high = high.rolling(n, min_periods=2).max().shift(1)
        range_low  = low.rolling(n, min_periods=2).min().shift(1)

        # Price breakout from prior range
        price_breaks_up   = close > range_high
        price_breaks_down = close < range_low
        price_breaks = price_breaks_up | price_breaks_down

        # Volume surge
        volume_surge = volume_ratio > _P["expansion_volume_ratio"]

        # Momentum increasing (RSI rising)
        momentum_rising = rsi.diff(mom_bars) > 0

        # Was recently compressed? Use 5-bar lookback on compression flag
        recently_compressed = is_compression.rolling(n, min_periods=1).max().shift(1).astype(bool)

        is_expansion = (
            price_breaks &
            volume_surge &
            momentum_rising &
            recently_compressed
        ).fillna(False)

        return pd.DataFrame({
            "is_expansion": is_expansion,
            "range_high":   range_high,
            "range_low":    range_low,
        })

    # ── DB Upsert ───────────────────────────────────────────────────────────

    def _upsert(self, session, records: list[dict]):
        """Batch upsert volatility regime records."""
        for i in range(0, len(records), 500):
            batch = records[i:i + 500]
            stmt = pg_insert(VolatilityRegime).values(batch)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_vol_regime_instrument_date",
                set_={
                    "atr_pct":               stmt.excluded.atr_pct,
                    "bb_width":              stmt.excluded.bb_width,
                    "realized_vol_20d":      stmt.excluded.realized_vol_20d,
                    "atr_percentile":        stmt.excluded.atr_percentile,
                    "bb_width_percentile":   stmt.excluded.bb_width_percentile,
                    "vol_percentile":        stmt.excluded.vol_percentile,
                    "regime":                stmt.excluded.regime,
                    "is_compression":        stmt.excluded.is_compression,
                    "is_expansion":          stmt.excluded.is_expansion,
                    "compression_range_high": stmt.excluded.compression_range_high,
                    "compression_range_low":  stmt.excluded.compression_range_low,
                },
            )
            session.execute(stmt)
        session.commit()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(val) -> Optional[float]:
        """Convert to float, return None if NaN/None."""
        if val is None:
            return None
        try:
            f = float(val)
            return round(f, 6) if not np.isnan(f) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _row_to_dict(row: VolatilityRegime) -> dict:
        return {
            "date":                  str(row.date),
            "regime":                row.regime,
            "is_compression":        row.is_compression,
            "is_expansion":          row.is_expansion,
            "atr_pct":               row.atr_pct,
            "bb_width":              row.bb_width,
            "realized_vol_20d":      row.realized_vol_20d,
            "atr_percentile":        row.atr_percentile,
            "bb_width_percentile":   row.bb_width_percentile,
            "vol_percentile":        row.vol_percentile,
            "compression_range_high": row.compression_range_high,
            "compression_range_low":  row.compression_range_low,
        }
