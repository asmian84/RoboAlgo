"""
DataCollector — fetches multi-timeframe OHLCV + options chain per symbol.

For each trade profile (SCALP/SWING/POSITION), fetches the required timeframes:
  SCALP:    5m (5d), 15m (10d), 1h (30d)
  SWING:    1h (60d), 4h (60d), 1d (120d)
  POSITION: 4h (60d), 1d (365d), 1wk (365d)

Options chain is fetched for symbols in the options focus list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import yfinance as yf

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


def collect(
    symbol: str,
    profile: str = "SWING",
    fetch_options: bool = False,
) -> SymbolData:
    """
    Fetch all required data for scoring a symbol under a given trade profile.

    Args:
        symbol: Ticker symbol (e.g., "AAPL")
        profile: Trade profile — "SCALP", "SWING", or "POSITION"
        fetch_options: Whether to also fetch options chain data

    Returns:
        SymbolData with multi-TF bars and optionally options chain
    """
    sd = SymbolData(symbol=symbol.upper(), profile=profile.upper())

    if sd.profile not in PROFILE_TFS:
        sd.error = f"Unknown profile: {sd.profile}"
        return sd

    ticker = yf.Ticker(sd.symbol)

    # ── Fetch multi-TF OHLCV bars ────────────────────────────────────────────
    tf_specs = PROFILE_TFS[sd.profile]

    for spec in tf_specs:
        tf = spec["tf"]
        period = spec["period"]
        try:
            # yfinance interval mapping
            # Note: yfinance doesn't support "4h" directly, use "60m" * 4
            yf_interval = tf
            if tf == "4h":
                # Fetch 1h and resample to 4h
                raw = ticker.history(period=period, interval="1h")
                if raw is not None and not raw.empty:
                    raw.columns = [c.lower() for c in raw.columns]
                    # Resample to 4h
                    ohlc_map = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                    df = raw.resample("4h").agg(ohlc_map).dropna()
                    sd.bars[tf] = df
                    logger.debug("%s %s: %d bars (resampled from 1h)", sd.symbol, tf, len(df))
            else:
                raw = ticker.history(period=period, interval=yf_interval)
                if raw is not None and not raw.empty:
                    raw.columns = [c.lower() for c in raw.columns]
                    # Keep only OHLCV columns
                    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
                    sd.bars[tf] = raw[cols]
                    logger.debug("%s %s: %d bars", sd.symbol, tf, len(raw))
        except Exception as e:
            logger.warning("Failed to fetch %s %s: %s", sd.symbol, tf, e)

    # ── Get current price ────────────────────────────────────────────────────
    # Use the most recent close from the shortest timeframe
    for tf_spec in tf_specs:
        tf = tf_spec["tf"]
        if tf in sd.bars and not sd.bars[tf].empty:
            sd.current_price = float(sd.bars[tf]["close"].iloc[-1])
            break

    # ── Fetch options chain (if requested) ───────────────────────────────────
    if fetch_options:
        try:
            # Try our existing options fetcher first
            from data_engine.yahoo_options_fetcher import OptionsChainFetcher
            chain = OptionsChainFetcher().fetch(sd.symbol)
            if chain:
                sd.options_chain = chain
                logger.debug("%s: options chain fetched", sd.symbol)
        except Exception as e:
            logger.debug("%s: no options data: %s", sd.symbol, e)

    # ── Basic info ───────────────────────────────────────────────────────────
    try:
        info = ticker.info
        sd.info = {
            "market_cap": info.get("marketCap", 0),
            "float_shares": info.get("floatShares", 0),
            "short_pct": info.get("shortPercentOfFloat", 0),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }
    except Exception:
        pass

    if not sd.bars:
        sd.error = f"No price data available for {sd.symbol}"

    return sd


def collect_batch(
    symbols: list[str],
    profile: str = "SWING",
    fetch_options_for: list[str] | None = None,
    max_workers: int = 4,
) -> list[SymbolData]:
    """
    Collect data for multiple symbols in parallel.

    Args:
        symbols: List of ticker symbols
        profile: Trade profile for all symbols
        fetch_options_for: Symbols that should also fetch options (e.g., OPTIONS_FOCUS)
        max_workers: Thread pool size

    Returns:
        List of SymbolData objects
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
                results.append(future.result(timeout=60))
            except Exception as e:
                sym = futures[future]
                sd = SymbolData(symbol=sym, profile=profile, error=str(e))
                results.append(sd)

    return results
