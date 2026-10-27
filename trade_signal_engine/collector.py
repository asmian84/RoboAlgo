"""
DataCollector — fetches multi-timeframe OHLCV + options chain per symbol.

For each trade profile (SCALP/SWING/POSITION), fetches the required timeframes:
  SCALP:    5m (5d), 15m (10d), 1h (30d), 1d (120d), 1wk (365d)
  SWING:    15m (10d), 1h (60d), 4h (60d), 1d (180d), 1wk (365d)
  POSITION: 4h (60d), 1d (365d), 1wk (365d)

SCALP and SWING carry daily and weekly bars because the scorer's stochastic setup gate
reads 1wk/1d/4h for confluence: without them the gate can never pass.

Data source priority:
  1. PostgreSQL DB (fastest — pre-cached daily bars)
  2. AlphaVantage (primary API — daily + intraday with rate limiting)
  3. yfinance (fallback only when AV unavailable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import threading

import pandas as pd

logger = logging.getLogger(__name__)

# Global lock for yfinance — it is NOT thread-safe.
# Without this, concurrent yf.download() calls return data for the wrong symbol.
_YF_LOCK = threading.Lock()

# ── Timeframe requirements per profile ────────────────────────────────────────

PROFILE_TFS = {
    "SCALP": [
        {"tf": "5m",  "period": "5d"},
        {"tf": "15m", "period": "10d"},
        {"tf": "1h",  "period": "30d"},
        {"tf": "1d",  "period": "120d"},
        {"tf": "1wk", "period": "365d"},
    ],
    "SWING": [
        {"tf": "15m", "period": "10d"},
        {"tf": "1h",  "period": "60d"},
        {"tf": "4h",  "period": "60d"},
        {"tf": "1d",  "period": "180d"},
        {"tf": "1wk", "period": "365d"},  # stoch setup can appear on any TF
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

    # Try 3: yfinance fallback (locked — yfinance is not thread-safe)
    try:
        import yfinance as yf
        with _YF_LOCK:
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


def _fetch_intraday_bars(symbol: str, interval: str, bars: int = 200, prefer_yfinance: bool = False) -> pd.DataFrame | None:
    """Fetch intraday OHLCV: AV first (if premium), then yfinance fallback.

    When prefer_yfinance=True (batch scanning), skip AV entirely to avoid rate limits.
    """
    # Try yfinance first when preferred (batch scan mode — no rate limits)
    if prefer_yfinance:
        try:
            import yfinance as yf
            period = "5d" if interval in ("5m",) else "10d" if interval == "15m" else "30d"
            with _YF_LOCK:
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

    # Try 2: yfinance fallback (locked — yfinance is not thread-safe)
    try:
        import yfinance as yf
        period = "5d" if interval in ("5m",) else "10d" if interval == "15m" else "30d"
        with _YF_LOCK:
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

def _fetch_tf_worker(symbol: str, tf: str, period: str, fast_mode: bool) -> tuple[str, pd.DataFrame | None]:
    """Fetch a single timeframe — designed to run in a thread pool."""
    bars = _PERIOD_TO_BARS.get(period, 200)
    ohlc_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    try:
        if tf == "1d":
            return tf, _fetch_daily_bars(symbol, bars=bars)
        elif tf == "1wk":
            raw = _fetch_daily_bars(symbol, bars=bars)
            if raw is not None and not raw.empty:
                weekly = raw.resample("W").agg(ohlc_map).dropna()
                return tf, (weekly if not weekly.empty else None)
            return tf, None
        elif tf == "4h":
            raw = _fetch_intraday_bars(symbol, "1h", bars=bars, prefer_yfinance=fast_mode)
            if raw is not None and not raw.empty:
                resampled = raw.resample("4h").agg(ohlc_map).dropna()
                return tf, (resampled if not resampled.empty else None)
            return tf, None
        else:
            return tf, _fetch_intraday_bars(symbol, tf, bars=bars, prefer_yfinance=fast_mode)
    except Exception as e:
        logger.warning("Failed to fetch %s %s: %s", symbol, tf, e)
        return tf, None


def collect(
    symbol: str,
    profile: str = "SWING",
    fetch_options: bool = False,
    fast_mode: bool = False,
) -> SymbolData:
    """
    Fetch all required data for scoring a symbol under a given trade profile.

    Data source priority: DB → AlphaVantage → yfinance (fallback).
    When fast_mode=True (batch scanning), use yfinance for all intraday to avoid AV rate limits.

    All timeframes are fetched IN PARALLEL using threads — each TF goes to a
    different source concurrently (DB for daily, yfinance for 15m, yfinance for 1h, etc.)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sd = SymbolData(symbol=symbol.upper(), profile=profile.upper())

    if sd.profile not in PROFILE_TFS:
        sd.error = f"Unknown profile: {sd.profile}"
        return sd

    tf_specs = PROFILE_TFS[sd.profile]

    # ── Parallel fetch all timeframes at once ────────────────────────────
    with ThreadPoolExecutor(max_workers=len(tf_specs)) as pool:
        futures = {
            pool.submit(_fetch_tf_worker, sd.symbol, spec["tf"], spec["period"], fast_mode): spec["tf"]
            for spec in tf_specs
        }
        for fut in as_completed(futures):
            tf, raw = fut.result()
            if raw is not None and not raw.empty:
                sd.bars[tf] = raw
                logger.debug("%s %s: %d bars", sd.symbol, tf, len(raw))

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


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH-FIRST DATA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Instead of downloading data one symbol at a time (500 symbols × 5 TFs =
#  2500 serial HTTP calls = 40+ minutes), we download ALL symbols at once
#  per timeframe in a single yf.download() call:
#
#    yf.download("AAPL MSFT NVDA ... 500 symbols", interval="1d")  → 3 seconds
#    yf.download("AAPL MSFT NVDA ... 500 symbols", interval="1h")  → 3 seconds
#    yf.download("AAPL MSFT NVDA ... 500 symbols", interval="15m") → 2 seconds
#
#  Total: ~10 seconds for ALL data vs 40+ minutes.
#  Then we slice the batch DataFrame per-symbol and build SymbolData objects.
# ═══════════════════════════════════════════════════════════════════════════════

# yfinance period mapping for each interval
_YF_INTERVAL_PERIOD = {
    "1d":  "1y",
    "1h":  "30d",
    "15m": "5d",
    "5m":  "5d",
}

# Max symbols per yfinance batch call (too many → timeout)
_YF_BATCH_SIZE = 200


def _batch_download_tf(
    symbols: list[str],
    interval: str,
    period: str,
    progress_callback: callable | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Batch-download one timeframe for ALL symbols at once.
    Returns {symbol: DataFrame} for symbols that got data.

    NOTE: No _YF_LOCK needed here — yf.download() with multiple symbols
    in a single call is safe. The lock was only needed when multiple threads
    called yf.download() with DIFFERENT single symbols simultaneously.
    Batch calls are atomic (one HTTP request for all symbols).

    Includes retry with backoff for Yahoo rate limits.
    """
    import time as _time
    result: dict[str, pd.DataFrame] = {}
    import yfinance as yf

    # Cap downloads: if >1000 symbols missing, only download first 600 to avoid
    # yfinance rate limits. The cache warmup fills the rest over time.
    _MAX_DOWNLOAD = 600
    if len(symbols) > _MAX_DOWNLOAD:
        logger.info("Capping %s download from %d to %d symbols (cache warmup handles rest)",
                     interval, len(symbols), _MAX_DOWNLOAD)
        symbols = symbols[:_MAX_DOWNLOAD]

    total_batches = (len(symbols) + _YF_BATCH_SIZE - 1) // _YF_BATCH_SIZE
    _consecutive_failures = 0
    for batch_num, i in enumerate(range(0, len(symbols), _YF_BATCH_SIZE)):
        batch = symbols[i:i + _YF_BATCH_SIZE]
        df = None

        # Bail early if yfinance is clearly rate-limiting us
        if _consecutive_failures >= 3:
            if progress_callback:
                progress_callback(f"Stopping {interval} downloads — rate limited after {len(result)} symbols")
            break

        if progress_callback:
            progress_callback(f"Downloading {interval}: batch {batch_num+1}/{total_batches} ({len(result)}/{len(symbols)} done)")

        # Retry up to 3 times with backoff for rate limits
        for attempt in range(3):
            try:
                df = yf.download(
                    " ".join(batch),
                    period=period,
                    interval=interval,
                    progress=False,
                    timeout=45,
                    group_by="ticker",
                    threads=True,
                )
                if df is not None and not df.empty:
                    break  # success
                # Empty result might be rate limit — retry
                if attempt < 2:
                    _time.sleep(2 * (attempt + 1))
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    logger.warning("Rate limited on %s batch %d, waiting %ds...",
                                   interval, i, 5 * (attempt + 1))
                    _time.sleep(5 * (attempt + 1))
                else:
                    logger.warning("Batch download %s chunk %d failed: %s", interval, i, e)
                    break

        if df is None or df.empty:
            _consecutive_failures += 1
            continue

        _consecutive_failures = 0  # reset on success
        for sym in batch:
            try:
                if len(batch) == 1:
                    sym_df = df.copy()
                else:
                    if sym not in df.columns.get_level_values(0):
                        continue
                    sym_df = df[sym].dropna(how="all")

                if sym_df.empty or len(sym_df) < 5:
                    continue

                # Flatten multi-level columns
                if isinstance(sym_df.columns, pd.MultiIndex):
                    sym_df.columns = sym_df.columns.get_level_values(0)
                sym_df.columns = [c.lower() for c in sym_df.columns]

                cols = [c for c in ["open", "high", "low", "close", "volume"] if c in sym_df.columns]
                if "close" not in cols:
                    continue
                result[sym] = sym_df[cols]
            except Exception:
                continue

    return result


# ── FMP Fallback Data Source ─────────────────────────────────────────────────
_FMP_API_KEY = "V981vaUJOuOmNrltZDCKNt3RSwsc0MGW"
_FMP_BASE = "https://financialmodelingprep.com/stable"

_FMP_INTERVAL_MAP = {
    "1d": ("historical-price-eod/full-stock-price", None),       # daily
    "1h": ("historical-price-full", "1hour"),      # 1h intraday
    "15m": ("historical-price-full", "15min"),     # 15m intraday
    "5m": ("historical-price-full", "5min"),       # 5m intraday
}


def _fmp_download_daily(symbols: list[str], progress_callback=None) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV from FMP for symbols yfinance failed on."""
    import requests
    result: dict[str, pd.DataFrame] = {}

    for idx, sym in enumerate(symbols):
        if progress_callback and idx % 20 == 0:
            progress_callback(f"FMP fallback: {idx}/{len(symbols)} daily bars")
        try:
            url = f"{_FMP_BASE}/historical-price-eod/full-stock-price"
            resp = requests.get(url, params={
                "symbol": sym,
                "apikey": _FMP_API_KEY,
            }, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data or not isinstance(data, list):
                continue

            rows = []
            for bar in data:
                rows.append({
                    "date": bar.get("date"),
                    "open": bar.get("open", 0),
                    "high": bar.get("high", 0),
                    "low": bar.get("low", 0),
                    "close": bar.get("close", 0),
                    "volume": bar.get("volume", 0),
                })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]]
            # Take last 365 days
            df = df.tail(365)
            if len(df) >= 20:
                result[sym] = df
        except Exception as e:
            logger.debug("FMP daily failed for %s: %s", sym, e)

    return result


def _fmp_download_intraday(symbols: list[str], interval: str, progress_callback=None) -> dict[str, pd.DataFrame]:
    """Download intraday OHLCV from FMP for symbols yfinance failed on."""
    import requests
    result: dict[str, pd.DataFrame] = {}

    fmp_interval_map = {"1h": "1hour", "15m": "15min", "5m": "5min"}
    fmp_tf = fmp_interval_map.get(interval)
    if not fmp_tf:
        return result

    for idx, sym in enumerate(symbols):
        if progress_callback and idx % 20 == 0:
            progress_callback(f"FMP fallback: {idx}/{len(symbols)} {interval} bars")
        try:
            url = f"{_FMP_BASE}/historical-chart/{fmp_tf}/{sym}"
            resp = requests.get(url, params={"apikey": _FMP_API_KEY}, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data or not isinstance(data, list):
                continue

            rows = []
            for bar in data:
                rows.append({
                    "date": bar.get("date"),
                    "open": bar.get("open", 0),
                    "high": bar.get("high", 0),
                    "low": bar.get("low", 0),
                    "close": bar.get("close", 0),
                    "volume": bar.get("volume", 0),
                })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]]
            if len(df) >= 10:
                result[sym] = df
        except Exception as e:
            logger.debug("FMP intraday %s failed for %s: %s", interval, sym, e)

    return result


def _resample_4h(hourly_df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample 1h bars to 4h."""
    ohlc_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    try:
        r = hourly_df.resample("4h").agg(ohlc_map).dropna()
        return r if not r.empty and len(r) >= 5 else None
    except Exception:
        return None


def _resample_weekly(daily_df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample daily bars to weekly."""
    ohlc_map = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    try:
        r = daily_df.resample("W").agg(ohlc_map).dropna()
        return r if not r.empty and len(r) >= 5 else None
    except Exception:
        return None


def batch_collect(
    symbols: list[str],
    profile: str = "SWING",
    progress_callback: callable | None = None,
    cache_only: bool = False,
) -> dict[str, SymbolData]:
    """
    Cache-first batch data collection.

    Priority order for each symbol+timeframe:
      1. Local parquet cache (~/.roboalgo/cache/) — instant, zero network
      2. PostgreSQL DB (daily bars only) — fast, reliable
      3. yfinance batch download — for anything missing from cache+DB

    Downloaded data is written back to the parquet cache so next scan is instant.
    A background updater refreshes the cache at whatever pace Yahoo allows.

    Returns {symbol: SymbolData} — ready for scoring.
    """
    import time as _time
    start = _time.time()
    profile = profile.upper()
    tf_specs = PROFILE_TFS.get(profile, PROFILE_TFS["SWING"])

    # Determine which raw intervals we need to download
    # (4h = resample from 1h, 1wk = resample from 1d)
    raw_intervals: dict[str, str] = {}  # interval -> period
    needs_4h = False
    needs_weekly = False

    for spec in tf_specs:
        tf = spec["tf"]
        if tf == "4h":
            needs_4h = True
            raw_intervals.setdefault("1h", "60d")  # need 1h to make 4h
        elif tf == "1wk":
            needs_weekly = True
            raw_intervals.setdefault("1d", "1y")    # need 1d to make weekly
        elif tf == "1d":
            period = _YF_INTERVAL_PERIOD.get("1d", "1y")
            raw_intervals.setdefault("1d", period)
        else:
            period = _YF_INTERVAL_PERIOD.get(tf, "30d")
            raw_intervals[tf] = period

    # ── Step 0: Read from local parquet cache (instant) ─────────────────
    # The cache is the source of truth — if it has data, use it.
    # Only go to network for symbols/TFs not in cache.
    from trade_signal_engine.data_cache import (
        read_cached_batch, write_cache, get_cache_age,
        trigger_background_update, _TF_DIRS,
    )

    # Map our raw intervals to cache TFs (cache uses 1d, 1h, 15m, 5m)
    _CACHE_TF_MAP = {"1d": "1d", "1h": "1h", "15m": "15m", "5m": "5m"}

    cached_data: dict[str, dict[str, pd.DataFrame]] = {}  # interval -> {sym -> df}
    cache_hits = 0
    cache_misses_by_tf: dict[str, list[str]] = {}  # interval -> [missing symbols]

    for interval in raw_intervals:
        cache_tf = _CACHE_TF_MAP.get(interval)
        if cache_tf and cache_tf in _TF_DIRS:
            cached = read_cached_batch(symbols, cache_tf)
            cached_data[interval] = cached
            cache_hits += len(cached)
            missing = [s for s in symbols if s not in cached]
            if missing:
                cache_misses_by_tf[interval] = missing
        else:
            cache_misses_by_tf[interval] = list(symbols)

    cache_age = get_cache_age()
    if cache_hits > 0:
        logger.info("Cache: %d hits across %d TFs (age: %.0fs)",
                     cache_hits, len(raw_intervals), cache_age)

    # Report cache results
    total_misses = sum(len(v) for v in cache_misses_by_tf.values())
    if progress_callback:
        if total_misses == 0:
            progress_callback(f"All {len(symbols)} symbols loaded from cache")
        else:
            progress_callback(f"Cache: {cache_hits} hits, {total_misses} misses to download")

    # ── Cache-only mode: skip all network calls (DB + yfinance) ────────
    if cache_only:
        cache_misses_by_tf = {}  # skip all downloads
        if progress_callback:
            progress_callback(f"Cache-only mode: {cache_hits} symbols available")

    # ── Step 1: DB daily bars for cache misses (instant, reliable) ──────
    db_daily: dict[str, pd.DataFrame] = {}
    daily_missing = cache_misses_by_tf.get("1d", []) if not cache_only else []

    if daily_missing and "1d" in raw_intervals:
        try:
            from database.connection import get_session
            from database.models import Instrument, PriceData
            from sqlalchemy import select, and_
            from datetime import date, timedelta

            start_date = date.today() - timedelta(days=400)
            session = get_session()
            try:
                id_map: dict[str, int] = {}
                rows = session.execute(
                    select(Instrument.symbol, Instrument.id).where(
                        Instrument.symbol.in_([s.upper() for s in daily_missing])
                    )
                ).fetchall()
                for sym, iid in rows:
                    id_map[sym] = iid

                if id_map:
                    price_rows = session.execute(
                        select(
                            Instrument.symbol,
                            PriceData.date,
                            PriceData.open,
                            PriceData.high,
                            PriceData.low,
                            PriceData.close,
                            PriceData.volume,
                        )
                        .join(Instrument, PriceData.instrument_id == Instrument.id)
                        .where(
                            and_(
                                Instrument.symbol.in_(list(id_map.keys())),
                                PriceData.date >= start_date,
                            )
                        )
                        .order_by(Instrument.symbol, PriceData.date)
                    ).fetchall()

                    df_all = pd.DataFrame(
                        price_rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
                    )
                    df_all["date"] = pd.to_datetime(df_all["date"])

                    for sym, grp in df_all.groupby("symbol"):
                        db_daily[sym] = grp.drop("symbol", axis=1).set_index("date")
            finally:
                session.close()

            logger.info("DB daily: %d/%d cache-missed symbols", len(db_daily), len(daily_missing))
            if progress_callback and db_daily:
                progress_callback(f"DB: loaded {len(db_daily)} symbols from database")
        except Exception as e:
            logger.debug("DB batch load failed: %s", e)

    # ── Step 2: yfinance batch for remaining misses ─────────────────────
    all_tf_data: dict[str, dict[str, pd.DataFrame]] = {}

    # Daily: cache → DB → yfinance
    if "1d" in raw_intervals:
        from_cache = cached_data.get("1d", {})
        # Symbols still missing after cache + DB
        still_missing = [s for s in daily_missing if s not in db_daily]
        yf_daily = _batch_download_tf(still_missing, "1d", raw_intervals["1d"], progress_callback=progress_callback) if still_missing else {}

        # Write new downloads to cache for next time
        for sym, df in {**db_daily, **yf_daily}.items():
            write_cache(sym, "1d", df)

        merged_daily = {**from_cache, **db_daily, **yf_daily}
        all_tf_data["1d"] = merged_daily
        logger.info("Daily: %d cache + %d DB + %d yfinance = %d total",
                     len(from_cache), len(db_daily), len(yf_daily), len(merged_daily))
    else:
        all_tf_data["1d"] = {**cached_data.get("1d", {}), **db_daily}

    # Intraday: cache → yfinance (no DB for intraday)
    intraday_intervals = {k: v for k, v in raw_intervals.items() if k != "1d"}
    if intraday_intervals:
        # Figure out which intervals actually need yfinance downloads
        intervals_needing_download = {}
        for interval, period in intraday_intervals.items():
            from_cache = cached_data.get(interval, {})
            if not cache_only:
                missing = [s for s in symbols if s not in from_cache]
                if missing:
                    intervals_needing_download[interval] = (period, missing)
            # Start with cache data
            all_tf_data[interval] = dict(from_cache)

        if intervals_needing_download and not cache_only:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(_batch_download_tf, missing_syms, interval, period, progress_callback): interval
                    for interval, (period, missing_syms) in intervals_needing_download.items()
                }
                for fut in as_completed(futures):
                    interval = futures[fut]
                    try:
                        data = fut.result()
                        # Write to cache
                        cache_tf = _CACHE_TF_MAP.get(interval, interval)
                        if cache_tf in _TF_DIRS:
                            for sym, df in data.items():
                                write_cache(sym, cache_tf, df)
                        # Merge with cached data
                        all_tf_data[interval].update(data)
                        logger.info("Batch %s: %d cached + %d downloaded",
                                    interval, len(cached_data.get(interval, {})), len(data))
                    except Exception as e:
                        logger.warning("Batch %s failed: %s", interval, e)
        else:
            logger.info("All intraday data served from cache — zero network calls")

    # Trigger background cache refresh if cache is getting stale (>5 min)
    if cache_age > 300:
        trigger_background_update(symbols, list(raw_intervals.keys()))

    # ── Step 3: Build SymbolData objects ─────────────────────────────────
    results: dict[str, SymbolData] = {}

    for sym in symbols:
        sd = SymbolData(symbol=sym, profile=profile)

        for spec in tf_specs:
            tf = spec["tf"]

            if tf == "1d":
                df = all_tf_data.get("1d", {}).get(sym)
                if df is not None and not df.empty:
                    sd.bars["1d"] = df

            elif tf == "1wk":
                daily = all_tf_data.get("1d", {}).get(sym)
                if daily is not None and not daily.empty:
                    weekly = _resample_weekly(daily)
                    if weekly is not None:
                        sd.bars["1wk"] = weekly

            elif tf == "4h":
                hourly = all_tf_data.get("1h", {}).get(sym)
                if hourly is not None and not hourly.empty:
                    fourh = _resample_4h(hourly)
                    if fourh is not None:
                        sd.bars["4h"] = fourh

            else:  # 5m, 15m, 1h — direct from batch
                df = all_tf_data.get(tf, {}).get(sym)
                if df is not None and not df.empty:
                    sd.bars[tf] = df

            # Also store raw 1h if we have it (even if not in profile TFs)
            if tf == "4h" and "1h" not in sd.bars:
                hourly = all_tf_data.get("1h", {}).get(sym)
                if hourly is not None and not hourly.empty:
                    sd.bars["1h"] = hourly

        # Set current price from most granular available TF
        for tf in ["5m", "15m", "1h", "4h", "1d", "1wk"]:
            if tf in sd.bars and not sd.bars[tf].empty and "close" in sd.bars[tf].columns:
                try:
                    sd.current_price = float(sd.bars[tf]["close"].iloc[-1])
                    break
                except (TypeError, ValueError, IndexError):
                    continue

        if not sd.bars:
            sd.error = f"No price data for {sym}"

        results[sym] = sd

    elapsed = _time.time() - start
    symbols_ok = sum(1 for sd in results.values() if not sd.error)
    logger.info("Batch collect complete: %d/%d symbols in %.1fs (%s profile, cache age: %.0fs)",
                symbols_ok, len(symbols), elapsed, profile, cache_age)

    return results
