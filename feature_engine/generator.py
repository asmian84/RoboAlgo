"""
RoboAlgo - Feature Generator
Converts indicators and price data into normalized feature vectors and stores them
in the features table.

Original features (v1):
  - trend_strength:        (close - MA200) / MA200
  - momentum:              RSI / 100
  - volatility_percentile: ATR / rolling_mean(ATR, 50)
  - volume_ratio:          volume / 20-day average volume
  - cycle_phase:           normalized 0.0–1.0 position in dominant cycle (from CycleMetric table)
  - macd_norm:             MACD line / close
  - bb_position:           (close - bb_lower) / (bb_upper - bb_lower)
  - price_to_ma50:         (close - MA50) / MA50
  - return_5d:             5-day return
  - return_20d:            20-day return

v2 factors (added from system audit):
  - momentum_acceleration: RSI 3-day rate-of-change, normalised -1→+1
      Formula: clip((RSI_t - RSI_{t-3}) / 3, -15, 15) / 15
      +1 = strongly accelerating upward momentum
      -1 = strongly decelerating (or reversing) momentum

  - volume_participation: directional volume quality, -1→+1
      Formula: direction × body_ratio × volume_quality
        direction    = sign(close - open)
        body_ratio   = |close - open| / (high - low + ε)
        volume_quality = min(volume / vol_20avg, 3.0) / 3.0
      +1 = high-participation bullish move, -1 = high-participation bearish move

  - correlation_exposure: 20-day rolling Pearson correlation to equal-weight
      portfolio of all tracked instruments. -1→+1.
      High positive = concentrated/correlated with rest of portfolio (risk)
      Near zero / negative = diversifying instrument
      Requires market_returns Series; falls back to None when not available.
"""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, Feature, CycleMetric

logger = logging.getLogger(__name__)

FEATURE_COLS_V1 = [
    "trend_strength", "momentum", "volatility_percentile", "volume_ratio",
    "cycle_phase", "macd_norm", "bb_position", "price_to_ma50",
    "return_5d", "return_20d",
]

FEATURE_COLS_V2 = [
    "momentum_acceleration",
    "volume_participation",
    "correlation_exposure",
]

FEATURE_COLS = FEATURE_COLS_V1 + FEATURE_COLS_V2


class FeatureGenerator:
    """Generates normalized feature vectors from price and indicator data."""

    # ── build_features ────────────────────────────────────────────────────────

    def build_features(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        market_returns: pd.Series | None = None,
        cycle_phases: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build feature matrix from prices and indicators.

        Args:
            prices:         OHLCV DataFrame indexed by date.
            indicators:     Indicator DataFrame indexed by date.
            market_returns: Optional equal-weight daily return Series indexed
                            by date, used to compute correlation_exposure.
            cycle_phases:   Optional Series indexed by date, values 0.0–1.0,
                            from CycleMetric.cycle_phase.  Falls back to 0.0
                            for dates with no cycle data.

        Returns:
            Feature DataFrame indexed by date.
        """
        df = prices.join(indicators, how="inner")
        if df.empty or len(df) < 200:
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # ── v1 features ───────────────────────────────────────────────────────

        # trend_strength = (close - MA200) / MA200
        features["trend_strength"] = (df["close"] - df["ma200"]) / df["ma200"]

        # momentum = RSI / 100 (normalized 0-1)
        features["momentum"] = df["rsi"] / 100.0

        # volatility_percentile = ATR / rolling_mean(ATR, 50)
        atr_rolling = df["atr"].rolling(50).mean()
        features["volatility_percentile"] = df["atr"] / atr_rolling

        # volume_ratio = volume / 20-day avg volume
        vol_ma20 = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / vol_ma20

        # cycle_phase: actual 0.0–1.0 position in dominant market cycle.
        # Source: CycleMetric.cycle_phase populated by the cycle engine.
        # Dates with no cycle metric row fall back to 0.5 (neutral mid-cycle).
        if cycle_phases is not None and not cycle_phases.empty:
            aligned = cycle_phases.reindex(df.index).ffill().bfill()
            features["cycle_phase"] = aligned.clip(0.0, 1.0).fillna(0.5)
        else:
            features["cycle_phase"] = 0.5   # neutral fallback (better than 0)

        # macd_norm = MACD line / close (price-normalized)
        features["macd_norm"] = df["macd_line"] / df["close"]

        # bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        bb_range = df["bb_upper"] - df["bb_lower"]
        features["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

        # price_to_ma50 = (close - MA50) / MA50
        features["price_to_ma50"] = (df["close"] - df["ma50"]) / df["ma50"]

        # returns
        features["return_5d"]  = df["close"].pct_change(5)
        features["return_20d"] = df["close"].pct_change(20)

        # ── v2 factors ────────────────────────────────────────────────────────

        # momentum_acceleration: RSI 3-day rate-of-change → −1 to +1
        # clip raw 3-day RSI delta at ±15 pts (extreme), divide to normalise
        rsi_delta = (df["rsi"] - df["rsi"].shift(3)) / 3.0
        features["momentum_acceleration"] = rsi_delta.clip(-15, 15) / 15.0

        # volume_participation: directional volume quality → −1 to +1
        hl_range    = (df["high"] - df["low"]).replace(0, np.nan)
        body        = df["close"] - df["open"]
        direction   = np.sign(body)
        body_ratio  = body.abs() / hl_range          # 0-1 (body as frac of range)
        vol_quality = (df["volume"] / vol_ma20.replace(0, np.nan)).clip(0, 3.0) / 3.0
        features["volume_participation"] = (direction * body_ratio * vol_quality).clip(-1, 1)

        # correlation_exposure: 20-day rolling Pearson corr with market portfolio
        if market_returns is not None and not market_returns.empty:
            inst_returns = df["close"].pct_change()
            # Align on shared dates
            aligned_market = market_returns.reindex(inst_returns.index)
            features["correlation_exposure"] = inst_returns.rolling(20).corr(aligned_market)
        else:
            features["correlation_exposure"] = np.nan

        # ── Drop warmup period ────────────────────────────────────────────────
        features = features.iloc[200:]
        features = features.replace([np.inf, -np.inf], np.nan)

        return features

    # ── compute_and_store ─────────────────────────────────────────────────────

    def compute_and_store(self, symbol: str | None = None):
        """Compute features for one or all instruments and store in the database."""
        session = get_session()
        try:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol)
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            # ── Pre-compute market returns (equal-weight portfolio) ──────────
            # Needed for correlation_exposure; computed once for the whole batch.
            market_returns = self._compute_market_returns(session, instruments)

            for instrument in tqdm(instruments, desc="Generating features"):
                self._process_instrument(session, instrument, market_returns)
        finally:
            session.close()

    # ── _compute_market_returns ───────────────────────────────────────────────

    @staticmethod
    def _compute_market_returns(session, instruments) -> pd.Series:
        """
        Build an equal-weight daily return series from all tracked instruments.
        Returns a pd.Series indexed by date (Timestamp) with dtype float.
        Falls back to an empty Series if fewer than 3 instruments are available.
        """
        if len(instruments) < 3:
            return pd.Series(dtype=float)

        close_frames = []
        for inst in instruments:
            rows = pd.read_sql(
                select(PriceData.date, PriceData.close)
                .where(PriceData.instrument_id == inst.id)
                .order_by(PriceData.date),
                session.bind,
            )
            if rows.empty:
                continue
            rows["date"] = pd.to_datetime(rows["date"])
            rows = rows.set_index("date")["close"].rename(inst.symbol)
            close_frames.append(rows)

        if not close_frames:
            return pd.Series(dtype=float)

        prices_wide = pd.concat(close_frames, axis=1)
        returns_wide = prices_wide.pct_change()
        # Equal-weight average return (ignore NaN columns per row)
        market_ret = returns_wide.mean(axis=1)
        return market_ret

    # ── _process_instrument ───────────────────────────────────────────────────

    def _process_instrument(self, session, instrument, market_returns: pd.Series):
        """Compute and store features for a single instrument."""
        # Load prices
        prices = pd.read_sql(
            select(PriceData.date, PriceData.open, PriceData.high,
                   PriceData.low, PriceData.close, PriceData.volume)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date),
            session.bind,
        )
        if prices.empty:
            return
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.set_index("date")

        # Load indicators
        indicators = pd.read_sql(
            select(Indicator.date, Indicator.rsi, Indicator.atr,
                   Indicator.macd_line, Indicator.macd_signal, Indicator.macd_histogram,
                   Indicator.bb_upper, Indicator.bb_middle, Indicator.bb_lower,
                   Indicator.bb_width, Indicator.ma50, Indicator.ma200)
            .where(Indicator.instrument_id == instrument.id)
            .order_by(Indicator.date),
            session.bind,
        )
        if indicators.empty:
            return
        indicators["date"] = pd.to_datetime(indicators["date"])
        indicators = indicators.set_index("date")

        # Load cycle phases from CycleMetric table
        cycle_phases_df = pd.read_sql(
            select(CycleMetric.date, CycleMetric.cycle_phase)
            .where(CycleMetric.instrument_id == instrument.id)
            .order_by(CycleMetric.date),
            session.bind,
        )
        if not cycle_phases_df.empty:
            cycle_phases_df["date"] = pd.to_datetime(cycle_phases_df["date"])
            cycle_phases = cycle_phases_df.set_index("date")["cycle_phase"].astype(float)
        else:
            cycle_phases = None

        features_df = self.build_features(prices, indicators, market_returns, cycle_phases)
        if features_df.empty:
            return

        # Build records
        records = []
        for dt, row in features_df.iterrows():
            record = {"instrument_id": instrument.id, "date": dt.date()}
            for col in FEATURE_COLS:
                val = row.get(col)
                record[col] = float(val) if pd.notna(val) else None
            records.append(record)

        # Batch upsert (conflict on the unique (instrument_id, date) constraint)
        update_cols = FEATURE_COLS  # update all columns including v2
        for i in range(0, len(records), 1000):
            batch = records[i:i + 1000]
            stmt = pg_insert(Feature).values(batch)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_feature_instrument_date",
                set_={col: stmt.excluded[col] for col in update_cols},
            )
            session.execute(stmt)

        session.commit()
        logger.info(f"Stored {len(records)} feature rows for {instrument.symbol}")

    # ── get_feature_matrix ────────────────────────────────────────────────────

    def get_feature_matrix(self, symbol: str) -> pd.DataFrame:
        """Load stored features for a symbol as a DataFrame."""
        session = get_session()
        try:
            instrument_id = session.execute(
                select(Instrument.id).where(Instrument.symbol == symbol)
            ).scalar()
            if instrument_id is None:
                return pd.DataFrame()

            df = pd.read_sql(
                select(Feature.date, *[getattr(Feature, c) for c in FEATURE_COLS])
                .where(Feature.instrument_id == instrument_id)
                .order_by(Feature.date),
                session.bind,
            )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        finally:
            session.close()
