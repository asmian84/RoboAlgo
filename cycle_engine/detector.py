"""
RoboAlgo - Cycle Detector
Detects dominant market cycles using spectral analysis (scipy.signal.periodogram).

Outputs per instrument per date:
  - cycle_length:  dominant cycle period in trading days
  - cycle_phase:   0.0-1.0 normalized position within the dominant cycle
  - cycle_strength: spectral power at the dominant frequency (normalized)
"""

import logging

import numpy as np
import pandas as pd
from scipy.signal import periodogram, detrend
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, CycleMetric

logger = logging.getLogger(__name__)

CYCLE_COLS = ["cycle_length", "cycle_phase", "cycle_strength"]


class CycleDetector:
    """Detects market cycles using spectral analysis."""

    def __init__(self, window: int = 120, min_cycle: int = 5, max_cycle: int = 60):
        """
        Args:
            window: Rolling window size for spectral analysis (trading days).
            min_cycle: Minimum cycle length to consider (days).
            max_cycle: Maximum cycle length to consider (days).
        """
        self.window = window
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle

    def detect_cycle(self, close: np.ndarray) -> tuple[float, float]:
        """Run periodogram on a price window to find the dominant cycle.

        Args:
            close: Array of close prices (length = self.window).

        Returns:
            (cycle_length, cycle_strength) tuple.
        """
        # Detrend the log prices to focus on oscillations
        log_prices = np.log(close)
        detrended = detrend(log_prices, type="linear")

        # Compute periodogram (frequency in cycles/day)
        freqs, power = periodogram(detrended, fs=1.0)

        # Filter to valid cycle range
        min_freq = 1.0 / self.max_cycle
        max_freq = 1.0 / self.min_cycle

        mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not mask.any():
            return 0.0, 0.0

        valid_freqs = freqs[mask]
        valid_power = power[mask]

        # Find dominant frequency
        peak_idx = np.argmax(valid_power)
        dominant_freq = valid_freqs[peak_idx]
        dominant_power = valid_power[peak_idx]

        cycle_length = 1.0 / dominant_freq if dominant_freq > 0 else 0.0

        # Normalize strength: ratio of peak power to total power
        total_power = valid_power.sum()
        cycle_strength = dominant_power / total_power if total_power > 0 else 0.0

        return cycle_length, cycle_strength

    def compute_phase(self, close: np.ndarray, cycle_length: float) -> float:
        """Estimate the current phase within the dominant cycle.

        Uses the last cycle_length bars to estimate where we are in the cycle
        via the Hilbert-like approach of fitting a sine wave.

        Args:
            close: Recent close prices.
            cycle_length: Dominant cycle length in days.

        Returns:
            Phase value between 0.0 and 1.0.
        """
        if cycle_length < 2:
            return 0.0

        n = min(len(close), int(cycle_length * 2))
        segment = np.log(close[-n:])
        segment = detrend(segment, type="linear")

        # Compute phase using the position relative to a fitted cycle
        t = np.arange(len(segment))
        freq = 2 * np.pi / cycle_length

        # Project onto sine and cosine basis
        sin_component = np.sum(segment * np.sin(freq * t))
        cos_component = np.sum(segment * np.cos(freq * t))

        phase = np.arctan2(sin_component, cos_component)
        # Normalize to 0-1
        return (phase + np.pi) / (2 * np.pi)

    def analyze_instrument(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Run rolling cycle detection on a price series.

        Args:
            prices: DataFrame with 'close' column, indexed by date.

        Returns:
            DataFrame with cycle_length, cycle_phase, cycle_strength columns.
        """
        close = prices["close"].values
        dates = prices.index

        if len(close) < self.window:
            return pd.DataFrame()

        results = []
        for i in range(self.window, len(close)):
            window_data = close[i - self.window:i]

            cycle_length, cycle_strength = self.detect_cycle(window_data)
            cycle_phase = self.compute_phase(window_data, cycle_length)

            results.append({
                "date": dates[i],
                "cycle_length": cycle_length,
                "cycle_phase": cycle_phase,
                "cycle_strength": cycle_strength,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("date")
        return df

    def compute_and_store(self, symbol: str | None = None):
        """Run cycle detection for one or all instruments and store in DB."""
        session = get_session()
        try:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol)
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            for instrument in tqdm(instruments, desc="Detecting cycles"):
                self._process_instrument(session, instrument)
        finally:
            session.close()

    def _process_instrument(self, session, instrument: Instrument):
        """Process a single instrument."""
        prices = pd.read_sql(
            select(PriceData.date, PriceData.close)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date),
            session.bind,
        )
        if prices.empty or len(prices) < self.window:
            return

        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.set_index("date")

        cycle_df = self.analyze_instrument(prices)
        if cycle_df.empty:
            return

        records = []
        for dt, row in cycle_df.iterrows():
            records.append({
                "instrument_id": instrument.id,
                "date": dt.date() if hasattr(dt, "date") else dt,
                "cycle_length": float(row["cycle_length"]),
                "cycle_phase": float(row["cycle_phase"]),
                "cycle_strength": float(row["cycle_strength"]),
            })

        for i in range(0, len(records), 1000):
            batch = records[i:i + 1000]
            stmt = pg_insert(CycleMetric).values(batch)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_cycle_instrument_date",
                set_={col: stmt.excluded[col] for col in CYCLE_COLS},
            )
            session.execute(stmt)

        session.commit()
        logger.info(f"Stored {len(records)} cycle metrics for {instrument.symbol}")
