"""
DataCollector — fetches multi-timeframe OHLCV + options chain per symbol.

For each trade profile (SCALP/SWING/POSITION), fetches the required timeframes:
  SCALP:    5m (5d), 15m (10d), 1h (30d)
  SWING:    1h (60d), 4h (60d), 1d (120d)
  POSITION: 4h (60d), 1d (365d), 1wk (365d)

Data source priority:
  1. PostgreSQL DB (fastest — pre-cached daily bars)
  2. AlphaVantage (primary API — daily + intraday with rate limiting)
  3. yfinance (fallback only when AV unavailable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Timeframe requirements per profile ────────────────────────────────────────

PROFILE_TFS = {
    "SCALP": [
        {"tf": "5m",  "period": "5d"},
        {"tf": "15m", "period": "10d"},
        {"tf": "1h",  "period": "30d"},
    ],
    "SWING": [
        {"tf": "1h",  "period": "60d"},
        {"tf": "4h",  "period": "60d"},
        {"tf": "1d",  "period": "120d"},
    ],
    "POSITION": [
        {"tf": "4h",  "period": "60d"},
        {"tf": "1d",  "period": "365d"},
        {"tf": "1wk", "period": "365d"},
    ],
}

# Map our timeframe labels to AV interval strings
_TF_TO_AV_INTERVAL = {
    "5m":  "5min",
    "15m": "15min",
    "1h":  "60min",
    # "4h" = resample from 60min
    # "1d" = daily
    # "1wk" = resample from daily
}

# Map period strings to lookback bar counts
_PERIOD_TO_BARS = {
    "5d": 80, "10d": 160, "30d": 200,
    "60d": 400, "120d": 500, "365d": 500, "1y": 500,
}


@dataclass
class SymbolData:
    """All data needed to score a single symbol."""
    symbol: str
    profile: str                              # SCALP | SWING | POSITION
    current_price: float = 0.0
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)  # tf -> OHLCV DataFrame
    options_chain: Any = None                 # OptionsChain from yahoo fetcher
    info: dict[str, Any] = field(default_factory=dict)  # yfinance info dict
    error: Optional[str] = None


# ── Data fetching helpers ─────────────────────────────────────────────────────

def _fetch_daily_bars(symbol: str, bars: int = 400) -> pd.DataFrame | None:
    """Fetch daily OHLCV: DB first, then AV, then yfinance fallback."""
    # Try 1: DB (fastest)
    try:
        from data_engine.alphavantage_fetcher import load_ohlcv_from_db
        result = load_ohlcv_from_db([symbol], lookback_days=bars)
        df = result.get(symbol)
        if df is not None and not df.empty and len(df) >= 10:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        logger.debug("DB fetch failed for %s: %s", symbol, e)

    # Try 2: AlphaVantage direct
    try:
        from data_engine.alphavantage_fetcher import fetch_daily_ohlcv
        df = fetch_daily_ohlcv(symbol, bars=bars)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        logger.debug("AV daily failed for %s: %s", symbol, e)

    # Try 3: yfinance fallback
    try:
        import yfinance as yf
        df = yf.download(symbol, period="1y", interval="1d", progress=False, timeout=15)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols]
    except Exception as e:
        logger.debug("yfinance daily failed for %s: %s", symbol, e)

    return None


def _fetch_intraday_bars(symbol: str, interval: str, bars: int = 200) -> pd.DataFrame | None:
    """Fetch intraday OHLCV: AV first (if premium), then yfinance fallback."""
    av_interval = _TF_TO_AV_INTERVAL.get(interval)

    # Try 1: AlphaVantage intraday
    if av_interval:
        try:
            from data_engine.alphavantage_fetcher import fetch_intraday_ohlcv, is_premium
            df = fetch_intraday_ohlcv(symbol, interval=av_interval, bars=bars)
            if df is not None and not df.empty and len(df) >= 10:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            logger.debug("AV intraday %s/%s failed: %s", symbol, interval, e)

    # Try 2: yfinance fallback
    try:
        import yfinance as yf
        # Map period
        period = "5d" if interval in ("5m",) else "10d" if interval == "15m" else "30d"
        df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=15)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            return df[cols]
    except Exception as e:
        logger.debug("yfinance intraday %s/%s failed: %s", symbol, interval, e)

    return None


# ── Main collector ────────────────────────────────────────────────────────────

def collect(
    symbol: str,
    profile: str = "SWING",
    fetch_options: bool = False,
) -> SymbolData:
    """
    Fetch all required data for scoring a symbol under a given trade profile.

    Data source priority: DB → AlphaVantage → yfinance (fallback).
    """
    sd = SymbolData(symbol=symbol.upper(), profile=profile.upper())

    if sd.profile not in PROFILE_TFS:
        sd.error = f"Unknown profile: {sd.profile}"
        return sd

    tf_specs = PROFILE_TFS[sd.profile]

    for spec in tf_specs:
        tf = spec["tf"]
        period = spec["period"]
        bars = _PERIOD_TO_BARS.get(period, 200)

        try:
            if tf == "1d":
                raw = _fetch_daily_bars(sd.symbol, bars=bars)
                if raw is not None and not raw.empty:
                    sd.bars[tf] = raw
                    logger.debug("%s %s: %d bars (daily)", sd.symbol, tf, len(raw))

            elif tf == "1wk":
                # Get daily and resample to weekly
                raw = _fetch_daily_bars(sd.symbol, bars=bars)
                if raw is not None and not raw.empty:
                    ohlc_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                    weekly = raw.resample("W").agg(ohlc_map).dropna()
                    if not weekly.empty:
                        sd.bars[tf] = weekly
                        logger.debug("%s %s: %d bars (weekly resample)", sd.symbol, tf, len(weekly))

            elif tf == "4h":
                # Get 1h and resample to 4h
                raw = _fetch_intraday_bars(sd.symbol, "1h", bars=bars)
                if raw is not None and not raw.empty:
                    ohlc_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                    resampled = raw.resample("4h").agg(ohlc_map).dropna()
                    if not resampled.empty:
                        sd.bars[tf] = resampled
                        logger.debug("%s %s: %d bars (4h resample)", sd.symbol, tf, len(resampled))

            else:
                # Direct intraday fetch (5m, 15m, 1h)
                raw = _fetch_intraday_bars(sd.symbol, tf, bars=bars)
                if raw is not None and not raw.empty:
                    sd.bars[tf] = raw
                    logger.debug("%s %s: %d bars", sd.symbol, tf, len(raw))

        except Exception as e:
            logger.warning("Failed to fetch %s %s: %s", sd.symbol, tf, e)

    # ── Get current price ────────────────────────────────────────────────────
    for tf_spec in tf_specs:
        tf = tf_spec["tf"]
        if tf in sd.bars and not sd.bars[tf].empty:
            sd.current_price = float(sd.bars[tf]["close"].iloc[-1])
            break

    # ── Fetch options chain (if requested) ───────────────────────────────────
    if fetch_options:
        try:
            from data_engine.yahoo_options_fetcher import OptionsChainFetcher
            chain = OptionsChainFetcher().fetch(sd.symbol)
            if chain:
                sd.options_chain = chain
                logger.debug("%s: options chain fetched", sd.symbol)
        except Exception as e:
            logger.debug("%s: no options data: %s", sd.symbol, e)

    if not sd.bars:
        sd.error = f"No price data available for {sd.symbol}"

    return sd


def collect_batch(
    symbols: list[str],
    profile: str = "SWING",
    fetch_options_for: list[str] | None = None,
    max_workers: int = 2,
) -> list[SymbolData]:
    """
    Collect data for multiple symbols with controlled concurrency.

    Uses max_workers=2 to respect AlphaVantage rate limits.
    """
    from concurrent.futures import ThreadPoolExecutor

    options_set = set(fetch_options_for or [])

    def _collect_one(sym: str) -> SymbolData:
        return collect(sym, profile=profile, fetch_options=sym in options_set)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_collect_one, sym): sym for sym in symbols}
        for future in futures:
            try:
                results.append(future.result(timeout=120))
            except Exception as e:
                sym = futures[future]
                sd = SymbolData(symbol=sym, profile=profile, error=str(e))
                results.append(sd)

    return results
