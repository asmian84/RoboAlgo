"""
RoboAlgo - Market Data Downloader
Downloads historical daily OHLCV data.
Primary source: Finnhub (if FINNHUB_API_KEY set in .env)
Fallback source: Yahoo Finance (yfinance)
"""

import logging
import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from config.settings import (
    ALL_INSTRUMENTS,
    DATA_START_DATE,
    DOWNLOAD_BATCH_SIZE,
    DOWNLOAD_DELAY_SECONDS,
    INDEX_DRIVERS,
    LEVERAGED_ETF_PAIRS,
    UNDERLYING_LEADERS,
)
from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# ── Finnhub client ────────────────────────────────────────────────────────────
_finnhub_client = None

def _get_finnhub():
    global _finnhub_client
    if _finnhub_client is not None:
        return _finnhub_client
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    try:
        import finnhub
        _finnhub_client = finnhub.Client(api_key=api_key)
        logger.info("Finnhub client initialised.")
    except Exception as e:
        logger.warning(f"Finnhub init failed: {e}")
        _finnhub_client = None
    return _finnhub_client


def _download_finnhub(symbol: str, start_date: str) -> pd.DataFrame | None:
    """Download OHLCV daily candles from Finnhub."""
    client = _get_finnhub()
    if client is None:
        return None
    try:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts   = int(datetime.now().timestamp())
        res = client.stock_candles(symbol, "D", start_ts, end_ts)
        if res.get("s") != "ok" or not res.get("c"):
            logger.debug(f"Finnhub: no data for {symbol} (status={res.get('s')})")
            return None
        df = pd.DataFrame({
            "date":   pd.to_datetime(res["t"], unit="s").dt.date,
            "open":   res["o"],
            "high":   res["h"],
            "low":    res["l"],
            "close":  res["c"],
            "volume": res["v"],
        })
        logger.info(f"Finnhub: {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        logger.warning(f"Finnhub error for {symbol}: {e}")
        return None


def _download_yfinance(symbol: str, start_date: str) -> pd.DataFrame | None:
    """Download OHLCV data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, auto_adjust=True)
        if df.empty:
            logger.warning(f"yfinance: no data for {symbol}")
            return None
        df = df.reset_index()
        col_map = {col: col.lower() for col in df.columns if isinstance(col, str)}
        df = df.rename(columns=col_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            logger.warning(f"yfinance: missing columns for {symbol}")
            return None
        logger.info(f"yfinance: {len(df)} rows for {symbol}")
        return df[list(required)]
    except Exception as e:
        logger.error(f"yfinance error for {symbol}: {e}")
        return None


class MarketDataDownloader:
    """Downloads and stores historical market data.

    Uses Finnhub as primary source (if FINNHUB_API_KEY in .env),
    falls back to Yahoo Finance automatically.
    """

    def __init__(self):
        self.instruments = ALL_INSTRUMENTS
        self.start_date = DATA_START_DATE
        self._use_finnhub = _get_finnhub() is not None

    def _ensure_instruments_exist(self, session):
        """Populate the instruments table with metadata for all configured instruments."""
        bull_map = {}
        bear_map = {}
        for bull, bear, desc, *_ in LEVERAGED_ETF_PAIRS:
            bull_map[bull] = (bear, desc)
            if bear:
                bear_map[bear] = (bull, desc)

        existing = set(session.execute(select(Instrument.symbol)).scalars())
        new_instruments = []

        for symbol in self.instruments:
            if symbol in existing:
                continue
            if symbol in INDEX_DRIVERS:
                new_instruments.append(Instrument(
                    symbol=symbol, name=symbol,
                    instrument_type="index", leverage_factor=1.0,
                    underlying=symbol,
                ))
            elif symbol in bull_map:
                pair, desc = bull_map[symbol]
                leverage = 3.0 if "3x" in desc else 2.0
                new_instruments.append(Instrument(
                    symbol=symbol, name=f"{desc} Bull",
                    instrument_type="leveraged_etf_bull",
                    leverage_factor=leverage,
                    underlying=desc, pair_symbol=pair,
                ))
            elif symbol in bear_map:
                pair, desc = bear_map[symbol]
                leverage = -3.0 if "3x" in desc else -2.0
                new_instruments.append(Instrument(
                    symbol=symbol, name=f"{desc} Bear",
                    instrument_type="leveraged_etf_bear",
                    leverage_factor=leverage,
                    underlying=desc, pair_symbol=pair,
                ))
            elif symbol in UNDERLYING_LEADERS:
                new_instruments.append(Instrument(
                    symbol=symbol, name=symbol,
                    instrument_type="stock", leverage_factor=1.0,
                    underlying=symbol,
                ))
            else:
                new_instruments.append(Instrument(
                    symbol=symbol, name=symbol,
                    instrument_type="unknown", leverage_factor=1.0,
                    underlying=symbol,
                ))

        if new_instruments:
            session.add_all(new_instruments)
            session.commit()
            logger.info(f"Created {len(new_instruments)} instrument records.")

    def _get_instrument_ids(self, session) -> dict[str, int]:
        rows = session.execute(select(Instrument.symbol, Instrument.id)).all()
        return {symbol: id_ for symbol, id_ in rows}

    def download_single(self, symbol: str) -> pd.DataFrame | None:
        """Download historical data. Tries Finnhub first, falls back to yfinance."""
        if self._use_finnhub:
            df = _download_finnhub(symbol, self.start_date)
            if df is not None and not df.empty:
                return df
            logger.debug(f"Finnhub miss for {symbol}, trying yfinance...")
        return _download_yfinance(symbol, self.start_date)

    def _store_prices(self, session, instrument_id: int, df: pd.DataFrame):
        records = []
        for _, row in df.iterrows():
            records.append({
                "instrument_id": instrument_id,
                "date":   row["date"],
                "open":   float(row["open"])   if pd.notna(row["open"])   else None,
                "high":   float(row["high"])   if pd.notna(row["high"])   else None,
                "low":    float(row["low"])    if pd.notna(row["low"])    else None,
                "close":  float(row["close"])  if pd.notna(row["close"])  else None,
                "volume": float(row["volume"]) if pd.notna(row["volume"]) else None,
            })
        if not records:
            return
        stmt = pg_insert(PriceData).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_price_instrument_date",
            set_={
                "open":   stmt.excluded.open,
                "high":   stmt.excluded.high,
                "low":    stmt.excluded.low,
                "close":  stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        session.execute(stmt)
        session.commit()

    def download_all(self, symbols: list[str] | None = None) -> dict:
        """Download data for all (or specified) instruments."""
        symbols = symbols or self.instruments
        session = get_session()
        source = "Finnhub+yfinance fallback" if self._use_finnhub else "yfinance"
        logger.info(f"Data source: {source}")
        try:
            self._ensure_instruments_exist(session)
            id_map = self._get_instrument_ids(session)
            success_count = 0
            fail_count = 0
            for i in tqdm(range(0, len(symbols), DOWNLOAD_BATCH_SIZE),
                         desc="Downloading market data"):
                batch = symbols[i:i + DOWNLOAD_BATCH_SIZE]
                for symbol in batch:
                    instrument_id = id_map.get(symbol)
                    if instrument_id is None:
                        logger.warning(f"No instrument record for {symbol}")
                        fail_count += 1
                        continue
                    df = self.download_single(symbol)
                    if df is not None and not df.empty:
                        self._store_prices(session, instrument_id, df)
                        success_count += 1
                    else:
                        fail_count += 1
                time.sleep(DOWNLOAD_DELAY_SECONDS)
            logger.info(f"Download complete: {success_count} ok, {fail_count} failed")
            return {"success": success_count, "failed": fail_count}
        finally:
            session.close()
