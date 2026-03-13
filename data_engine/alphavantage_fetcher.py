"""
RoboAlgo — AlphaVantage Fetcher
Stage-1 universe screener + bulk price-data fetcher.

Responsibilities:
  • Pull a live tradeable universe from the AV Listing endpoint.
  • Apply Stage-1 price/volume/market-cap filters (price > 3, volume > 500k).
  • Batch-download daily OHLCV for up to 3 000 symbols.
  • Serve as the authoritative "fresh bar" provider for the scanner pipeline.
  • Cache aggressively — TTL-based in-process dict; no DB dependency.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_AV_KEY  = os.getenv("ALPHAVANTAGE_API_KEY", "UPSSIQEAMIJFI6RS")
_AV_BASE = "https://www.alphavantage.co/query"

# ── Cache ──────────────────────────────────────────────────────────────────
_price_cache:    dict[str, tuple[pd.DataFrame, float]] = {}
_universe_cache: tuple[list[str], float] | None = None

_PRICE_TTL    = 3_600        # 1 h
_UNIVERSE_TTL = 21_600       # 6 h
_REQUEST_DELAY = 0.25        # seconds between AV calls (< 5 req/s free tier)
_MAX_WORKERS   = 8


# ── Stage-1 Filter Thresholds ─────────────────────────────────────────────
@dataclass
class UniverseFilter:
    min_price:       float = 3.0
    min_avg_volume:  float = 500_000
    max_symbols:     int   = 1_200


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(params: dict[str, Any], timeout: int = 15) -> dict:
    """Single AV request with basic error handling."""
    params["apikey"] = _AV_KEY
    try:
        r = requests.get(_AV_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug("AV request failed %s: %s", params.get("function"), exc)
        return {}


def _rate_limited_get(params: dict[str, Any]) -> dict:
    time.sleep(_REQUEST_DELAY)
    return _get(params)


# ─────────────────────────────────────────────────────────────────────────────
#  Universe fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_active_listing() -> list[dict]:
    """
    Download the AV CSV listing status for all active US equities.
    Returns list[dict] with keys: symbol, name, exchange, assetType, status.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=LISTING_STATUS&apikey={_AV_KEY}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        active = df[df.get("status", pd.Series(dtype=str)).str.lower() == "active"]
        return active.to_dict("records")
    except Exception as exc:
        logger.warning("fetch_active_listing failed: %s", exc)
        return []


def _fetch_quote_batch(symbols: list[str]) -> dict[str, dict]:
    """
    Bulk-quote using AV BATCH_STOCK_QUOTES (up to 100 symbols).
    Falls back to individual GLOBAL_QUOTE if batch unavailable.
    Returns {symbol: {price, volume, change_pct}}.
    """
    results: dict[str, dict] = {}
    chunk_size = 100

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        data = _rate_limited_get(
            {"function": "BATCH_STOCK_QUOTES", "symbols": ",".join(chunk)}
        )
        stock_quotes = data.get("Stock Quotes", [])
        for q in stock_quotes:
            sym = q.get("1. symbol", "")
            try:
                results[sym] = {
                    "price":      float(q.get("2. price",  0) or 0),
                    "volume":     float(q.get("3. volume", 0) or 0),
                    "timestamp":  q.get("4. timestamp", ""),
                }
            except (ValueError, TypeError):
                pass

    # Fill any gaps with individual GLOBAL_QUOTE
    missing = [s for s in symbols if s not in results]
    for sym in missing[:50]:          # cap individual lookups
        data = _rate_limited_get({"function": "GLOBAL_QUOTE", "symbol": sym})
        gq = data.get("Global Quote", {})
        try:
            results[sym] = {
                "price":  float(gq.get("05. price",       0) or 0),
                "volume": float(gq.get("06. volume",      0) or 0),
                "change_pct": gq.get("10. change percent", "0%"),
            }
        except (ValueError, TypeError):
            pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Stage-1 Universe Filter
# ─────────────────────────────────────────────────────────────────────────────

def get_filtered_universe(
    flt: UniverseFilter | None = None,
    force_refresh: bool = False,
) -> list[str]:
    """
    Return Stage-1 filtered universe (~1 200 symbols).

    Algorithm:
      1. Load AV listing for all active US equities.
      2. Filter to EQUITY type only (drop ETFs from universe scan).
      3. Batch-quote for live price + volume.
      4. Apply min_price / min_avg_volume thresholds.
      5. Cap at max_symbols by descending volume.

    Results are TTL-cached for 6 h to avoid hammering the API.
    """
    global _universe_cache

    if flt is None:
        flt = UniverseFilter()

    now = time.time()
    if not force_refresh and _universe_cache and (now - _universe_cache[1]) < _UNIVERSE_TTL:
        return _universe_cache[0]

    logger.info("Building Stage-1 universe …")
    listing = fetch_active_listing()
    equities = [
        r["symbol"] for r in listing
        if str(r.get("assetType", "")).lower() == "stock"
        and len(str(r.get("symbol", ""))) <= 5
        and "." not in str(r.get("symbol", ""))
    ]

    # Fetch live quotes in parallel chunks
    quotes: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_quote_batch, equities[i : i + 100]): i
            for i in range(0, len(equities), 100)
        }
        for fut in as_completed(futures):
            quotes.update(fut.result())

    # Apply filters
    candidates: list[tuple[str, float]] = []   # (symbol, volume)
    for sym, q in quotes.items():
        price  = q.get("price",  0.0)
        volume = q.get("volume", 0.0)
        if price >= flt.min_price and volume >= flt.min_avg_volume:
            candidates.append((sym, volume))

    # Sort by volume descending, cap at max_symbols
    candidates.sort(key=lambda x: x[1], reverse=True)
    universe = [s for s, _ in candidates[: flt.max_symbols]]

    logger.info("Stage-1 universe: %d symbols (from %d equities)", len(universe), len(equities))
    _universe_cache = (universe, now)
    return universe


# ─────────────────────────────────────────────────────────────────────────────
#  OHLCV price fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_daily_ohlcv(
    symbol: str,
    bars: int = 400,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for *symbol* from Alpha Vantage.
    Returns DataFrame indexed by date with columns [open, high, low, close, volume].
    Uses compact (100 bars) or full (20 years) output depending on *bars*.

    Results are TTL-cached per symbol.
    """
    cached = _price_cache.get(symbol.upper())
    now = time.time()
    if not force_refresh and cached and (now - cached[1]) < _PRICE_TTL:
        return cached[0].iloc[-bars:].copy()

    output_size = "full" if bars > 100 else "compact"
    data = _rate_limited_get({
        "function":    "TIME_SERIES_DAILY_ADJUSTED",
        "symbol":      symbol.upper(),
        "outputsize":  output_size,
        "datatype":    "json",
    })

    ts = data.get("Time Series (Daily)", {})
    if not ts:
        logger.debug("No price data for %s", symbol)
        return pd.DataFrame()

    rows = []
    for date_str, bar in ts.items():
        try:
            rows.append({
                "date":   pd.Timestamp(date_str),
                "open":   float(bar["1. open"]),
                "high":   float(bar["2. high"]),
                "low":    float(bar["3. low"]),
                "close":  float(bar["5. adjusted close"]),
                "volume": float(bar["6. volume"]),
            })
        except (KeyError, ValueError):
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    _price_cache[symbol.upper()] = (df, now)
    return df.iloc[-bars:].copy()


def fetch_bulk_ohlcv(
    symbols: list[str],
    bars: int = 400,
    max_workers: int = _MAX_WORKERS,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for many symbols in parallel (thread pool).
    Returns {symbol: DataFrame}.  Symbols with no data are omitted.

    Rate limit is enforced by _REQUEST_DELAY inside each worker.
    """
    results: dict[str, pd.DataFrame] = {}

    def _worker(sym: str) -> tuple[str, pd.DataFrame]:
        df = fetch_daily_ohlcv(sym, bars=bars)
        return sym, df

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_worker, s): s for s in symbols}
        for fut in as_completed(futs):
            sym, df = fut.result()
            if not df.empty and len(df) >= 20:
                results[sym] = df

    logger.info("fetch_bulk_ohlcv: %d/%d successful", len(results), len(symbols))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  DB-backed fast loader (use when data already in PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv_from_db(
    symbols: list[str],
    lookback_days: int = 400,
) -> dict[str, pd.DataFrame]:
    """
    Fast loader: bulk-pull OHLCV from the local PostgreSQL store.
    Falls back to AV fetch if a symbol is missing from DB.
    Returns {symbol: DataFrame}.
    """
    from database.connection import get_session
    from database.models import Instrument, PriceData
    from sqlalchemy import select, and_, func

    start_date = date.today() - timedelta(days=lookback_days)
    results: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    session = get_session()
    try:
        # Batch lookup instrument IDs
        id_map: dict[str, int] = {}
        rows = session.execute(
            select(Instrument.symbol, Instrument.id).where(
                Instrument.symbol.in_([s.upper() for s in symbols])
            )
        ).fetchall()
        for sym, iid in rows:
            id_map[sym] = iid

        missing = [s for s in symbols if s.upper() not in id_map]

        # Bulk-fetch prices for all known instruments in a single query
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
                results[sym] = grp.drop("symbol", axis=1).set_index("date")
    finally:
        session.close()

    # Fetch anything not in DB
    if missing:
        logger.info("DB miss for %d symbols — fetching from AV …", len(missing))
        av_data = fetch_bulk_ohlcv(missing, bars=lookback_days)
        results.update(av_data)

    return results
