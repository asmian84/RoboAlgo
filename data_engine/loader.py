"""
RoboAlgo - Data Loader
Retrieves price, indicator, and feature data from PostgreSQL into DataFrames.
"""

import logging
from datetime import date

import pandas as pd
from sqlalchemy import select, and_, func

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads data from PostgreSQL into pandas DataFrames."""

    def get_instrument_id(self, symbol: str) -> int | None:
        """Look up instrument_id by symbol."""
        session = get_session()
        try:
            return session.execute(
                select(Instrument.id).where(Instrument.symbol == symbol)
            ).scalar()
        finally:
            session.close()

    def get_prices(self, symbol: str, start_date: date | None = None,
                   end_date: date | None = None) -> pd.DataFrame:
        """Load OHLCV price data for a symbol."""
        session = get_session()
        try:
            instrument_id = self.get_instrument_id(symbol)
            if instrument_id is None:
                return pd.DataFrame()

            conditions = [PriceData.instrument_id == instrument_id]
            if start_date:
                conditions.append(PriceData.date >= start_date)
            if end_date:
                conditions.append(PriceData.date <= end_date)

            stmt = (
                select(PriceData.date, PriceData.open, PriceData.high,
                       PriceData.low, PriceData.close, PriceData.volume)
                .where(and_(*conditions))
                .order_by(PriceData.date)
            )

            df = pd.read_sql(stmt, session.bind)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        finally:
            session.close()

    def get_indicators(self, symbol: str) -> pd.DataFrame:
        """Load indicator data for a symbol."""
        session = get_session()
        try:
            instrument_id = self.get_instrument_id(symbol)
            if instrument_id is None:
                return pd.DataFrame()

            stmt = (
                select(
                    Indicator.date, Indicator.rsi, Indicator.atr,
                    Indicator.macd_line, Indicator.macd_signal, Indicator.macd_histogram,
                    Indicator.bb_upper, Indicator.bb_middle, Indicator.bb_lower,
                    Indicator.bb_width, Indicator.ma50, Indicator.ma200,
                )
                .where(Indicator.instrument_id == instrument_id)
                .order_by(Indicator.date)
            )

            df = pd.read_sql(stmt, session.bind)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        finally:
            session.close()

    def get_all_symbols(self) -> list[str]:
        """Return all instrument symbols."""
        session = get_session()
        try:
            return list(
                session.execute(
                    select(Instrument.symbol).order_by(Instrument.symbol)
                ).scalars()
            )
        finally:
            session.close()

    def get_price_count(self, symbol: str) -> int:
        """Count price records for a symbol."""
        session = get_session()
        try:
            instrument_id = self.get_instrument_id(symbol)
            if instrument_id is None:
                return 0
            return session.execute(
                select(func.count(PriceData.id)).where(
                    PriceData.instrument_id == instrument_id
                )
            ).scalar() or 0
        finally:
            session.close()
