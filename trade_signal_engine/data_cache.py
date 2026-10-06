# trade_signal_engine/data_cache.py
"""
Local OHLCV Data Cache — the source of truth for the scanner.

Instead of downloading from Yahoo on every scan (slow, rate-limited),
we maintain a local parquet cache that the scanner reads instantly.

A background updater refreshes the cache at whatever pace Yahoo allows:
  - Fresh cache (<5 min): scanner reads directly, zero network calls
  - Stale cache (>5 min): scanner reads stale + triggers background refresh
  - No cache: blocks briefly to do initial download, then caches

Cache structure:
  ~/.roboalgo/cache/
    daily/AAPL.parquet     — 5y of daily OHLCV
    hourly/AAPL.parquet    — 30d of 1h OHLCV
    15min/AAPL.parquet     — 5d of 15m OHLCV
    meta.json              — last update timestamps per symbol
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Cache config ─────────────────────────────────────────────────────────────

_CACHE_DIR = Path.home() / ".roboalgo" / "cache"
_META_FILE = _CACHE_DIR / "meta.json"
_CACHE_TTL = 300  # 5 min — cache is "fresh" for this long
_STALE_OK = 3600  # 1 hour — stale cache is still usable (better than nothing)

_TF_DIRS = {
    "1d":  "daily",
    "1h":  "hourly",
    "15m": "15min",
    "5m":  "5min",
}

_TF_PERIODS = {
    "1d":  "5y",
    "1h":  "30d",
    "15m": "5d",
    "5m":  "5d",
}

# In-memory cache for hot reads (avoids parquet I/O on every scan)
_MEM_CACHE: dict[str, dict[str, pd.DataFrame]] = {}  # tf -> {sym -> df}
_MEM_TS: dict[str, float] = {}  # tf -> timestamp of last load

# Background updater lock
_UPDATE_LOCK = threading.Lock()
_updating = False


def _ensure_dirs():
    """Create cache directories if they don't exist."""
    for subdir in _TF_DIRS.values():
        (_CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def _load_meta() -> dict:
    """Load cache metadata (last update timestamps)."""
    if _META_FILE.exists():
        try:
            return json.loads(_META_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_meta(meta: dict):
    """Save cache metadata."""
    _ensure_dirs()
    _META_FILE.write_text(json.dumps(meta, indent=2))


def _cache_path(tf: str, symbol: str) -> Path:
    """Get the parquet file path for a symbol+timeframe."""
    subdir = _TF_DIRS.get(tf, tf)
    return _CACHE_DIR / subdir / f"{symbol}.parquet"


def read_cached(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Read cached OHLCV data for a symbol. Returns None if not cached."""
    # Check in-memory first
    mem = _MEM_CACHE.get(tf, {}).get(symbol)
    if mem is not None:
        return mem

    # Check disk
    path = _cache_path(tf, symbol)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            # Store in memory for fast re-reads
            _MEM_CACHE.setdefault(tf, {})[symbol] = df
            return df
        except Exception:
            pass
    return None


def read_cached_batch(symbols: list[str], tf: str) -> dict[str, pd.DataFrame]:
    """Read cached data for multiple symbols. Returns {symbol: df}."""
    result = {}
    for sym in symbols:
        df = read_cached(sym, tf)
        if df is not None and not df.empty:
            result[sym] = df
    return result


def write_cache(symbol: str, tf: str, df: pd.DataFrame):
    """Write OHLCV data to cache."""
    if df is None or df.empty:
        return
    _ensure_dirs()
    path = _cache_path(tf, symbol)
    try:
        df.to_parquet(path)
        # Update in-memory cache
        _MEM_CACHE.setdefault(tf, {})[symbol] = df
    except Exception as e:
        logger.debug("Cache write failed for %s/%s: %s", symbol, tf, e)


def update_cache(
    symbols: list[str],
    timeframes: list[str] | None = None,
    batch_size: int = 100,
    max_rate_retries: int = 3,
) -> dict[str, dict[str, int]]:
    """
    Download fresh data from Yahoo and update the local cache.

    Downloads in batches to respect rate limits.
    Returns stats: {tf: {"downloaded": N, "cached": N, "failed": N}}
    """
    global _updating
    import yfinance as yf

    if timeframes is None:
        timeframes = list(_TF_DIRS.keys())

    _ensure_dirs()
    stats: dict[str, dict[str, int]] = {}

    with _UPDATE_LOCK:
        _updating = True
        try:
            for tf in timeframes:
                period = _TF_PERIODS.get(tf, "30d")
                downloaded = 0
                failed = 0

                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i:i + batch_size]
                    df = None

                    for attempt in range(max_rate_retries):
                        try:
                            df = yf.download(
                                " ".join(batch),
                                period=period,
                                interval=tf,
                                progress=False,
                                timeout=45,
                                group_by="ticker",
                                threads=True,
                            )
                            if df is not None and not df.empty:
                                break
                            if attempt < max_rate_retries - 1:
                                time.sleep(3 * (attempt + 1))
                        except Exception as e:
                            if attempt < max_rate_retries - 1:
                                time.sleep(5 * (attempt + 1))
                            else:
                                logger.warning("Cache update %s batch %d failed: %s", tf, i, e)

                    if df is None or df.empty:
                        failed += len(batch)
                        continue

                    # Parse and cache each symbol
                    for sym in batch:
                        try:
                            if len(batch) == 1:
                                sym_df = df.copy()
                            else:
                                if sym not in df.columns.get_level_values(0):
                                    failed += 1
                                    continue
                                sym_df = df[sym].dropna(how="all")

                            if sym_df.empty or len(sym_df) < 5:
                                failed += 1
                                continue

                            if isinstance(sym_df.columns, pd.MultiIndex):
                                sym_df.columns = sym_df.columns.get_level_values(0)
                            sym_df.columns = [c.lower() for c in sym_df.columns]

                            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in sym_df.columns]
                            if "close" not in cols:
                                failed += 1
                                continue

                            write_cache(sym, tf, sym_df[cols])
                            downloaded += 1
                        except Exception:
                            failed += 1

                    # Small delay between batches to be nice to Yahoo
                    time.sleep(0.5)

                stats[tf] = {"downloaded": downloaded, "failed": failed}
                logger.info("Cache update %s: %d downloaded, %d failed",
                            tf, downloaded, failed)

            # Update metadata
            meta = _load_meta()
            meta["last_update"] = time.time()
            meta["symbols_count"] = len(symbols)
            _save_meta(meta)

        finally:
            _updating = False

    return stats


def trigger_background_update(symbols: list[str], timeframes: list[str] | None = None):
    """Start a background cache update (non-blocking)."""
    if _updating:
        logger.info("Cache update already in progress, skipping")
        return

    thread = threading.Thread(
        target=update_cache,
        args=(symbols, timeframes),
        daemon=True,
    )
    thread.start()
    logger.info("Background cache update started for %d symbols", len(symbols))


def get_cache_age() -> float:
    """Return seconds since last cache update, or infinity if never updated."""
    meta = _load_meta()
    last = meta.get("last_update", 0)
    if last == 0:
        return float("inf")
    return time.time() - last


def get_cache_stats() -> dict:
    """Return cache statistics."""
    meta = _load_meta()
    stats = {
        "last_update": meta.get("last_update", 0),
        "age_seconds": get_cache_age(),
        "symbols_count": meta.get("symbols_count", 0),
        "updating": _updating,
    }
    # Count cached files
    for tf, subdir in _TF_DIRS.items():
        d = _CACHE_DIR / subdir
        if d.exists():
            stats[f"{tf}_cached"] = len(list(d.glob("*.parquet")))
        else:
            stats[f"{tf}_cached"] = 0
    return stats
