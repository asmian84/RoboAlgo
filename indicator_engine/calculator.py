"""
RoboAlgo - Indicator Calculator
Computes technical indicators from price data and stores them in PostgreSQL.

Indicators: RSI(14), ATR(14), MACD(12/26/9), Bollinger Bands(20,2), SMA 50, SMA 200.
"""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from config.settings import INDICATOR_PARAMS
from database.connection import get_session
from database.models import Instrument, PriceData, Indicator

logger = logging.getLogger(__name__)

# Column names in the Indicator model
INDICATOR_COLS = [
    "rsi", "atr", "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "ma50", "ma200",
]


class IndicatorCalculator:
    """Calculates technical indicators from OHLCV price data."""

    def __init__(self):
        self.params = INDICATOR_PARAMS

    def compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
        """Average True Range — Wilder EMA smoothing (alpha = 1/period).

        Intentional pandas adapter: matches indicator_engine.technical.atr()
        but operates on pandas Series for bulk DB storage pipelines.
        For hot-path scanning use indicator_engine.technical.atr() directly.
        """
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    def compute_macd(self, close: pd.Series, fast: int = 12, slow: int = 26,
                     signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, and histogram."""
        ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    def compute_bollinger(self, close: pd.Series, period: int = 20,
                          std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: upper, middle, lower, width.

        Intentional pandas adapter: mirrors indicator_engine.technical.bollinger()
        for bulk DB storage pipelines.  Uses pandas rolling().std() (sample std,
        ddof=1) which matches industry convention and existing stored data.
        For hot-path scanning use indicator_engine.technical.bollinger() directly.
        """
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        width = (upper - lower) / middle
        return upper, middle, lower, width

    def compute_sma(self, close: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return close.rolling(window=period, min_periods=period).mean()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators from OHLCV DataFrame.

        Args:
            df: DataFrame with columns open, high, low, close, volume (DatetimeIndex).

        Returns:
            DataFrame with indicator columns, indexed by date.
        """
        p = self.params
        close, high, low = df["close"], df["high"], df["low"]

        indicators = pd.DataFrame(index=df.index)
        indicators["rsi"] = self.compute_rsi(close, p["rsi_period"])
        indicators["atr"] = self.compute_atr(high, low, close, p["atr_period"])

        macd_line, macd_signal, macd_hist = self.compute_macd(
            close, p["macd_fast"], p["macd_slow"], p["macd_signal"]
        )
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_hist

        bb_upper, bb_middle, bb_lower, bb_width = self.compute_bollinger(
            close, p["bb_period"], p["bb_std"]
        )
        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_middle
        indicators["bb_lower"] = bb_lower
        indicators["bb_width"] = bb_width

        indicators["ma50"] = self.compute_sma(close, p["sma_short"])
        indicators["ma200"] = self.compute_sma(close, p["sma_long"])

        return indicators

    def compute_and_store(self, symbol: str | None = None):
        """Compute indicators for one or all instruments and store in the database."""
        session = get_session()
        try:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol)
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            for instrument in tqdm(instruments, desc="Computing indicators"):
                self._process_instrument(session, instrument)
        finally:
            session.close()

    def _process_instrument(self, session, instrument: Instrument):
        """Compute and store indicators for a single instrument."""
        stmt = (
            select(PriceData.date, PriceData.open, PriceData.high,
                   PriceData.low, PriceData.close, PriceData.volume)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        )
        df = pd.read_sql(stmt, session.bind)

        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {instrument.symbol} ({len(df)} rows)")
            return

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        indicators_df = self.calculate_all(df)
        indicators_df = indicators_df.dropna(how="all")
        if indicators_df.empty:
            return

        records = []
        for dt, row in indicators_df.iterrows():
            record = {"instrument_id": instrument.id, "date": dt.date()}
            for col in INDICATOR_COLS:
                val = row.get(col)
                record[col] = float(val) if pd.notna(val) else None
            records.append(record)

        # Batch upsert
        for i in range(0, len(records), 1000):
            batch = records[i:i + 1000]
            stmt = pg_insert(Indicator).values(batch)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_indicator_instrument_date",
                set_={col: stmt.excluded[col] for col in INDICATOR_COLS},
            )
            session.execute(stmt)

        session.commit()
        logger.info(f"Stored {len(records)} indicator rows for {instrument.symbol}")
