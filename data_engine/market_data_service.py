"""
data_engine/market_data_service.py
────────────────────────────────────────────────────────────────────────────────
Real-time market data service with multi-source fallback and in-memory caching.

Priority:
  Primary  → yfinance  (free, reliable for most tickers)
  Secondary → Finnhub   (rate-limited but broader real-time coverage)

Output contract (all sources normalized to):
  {
    "symbol":         str,
    "price":          float,
    "open":           float,
    "high":           float,
    "low":            float,
    "volume":         int,
    "avg_volume":     int,
    "change":         float,      # absolute
    "change_percent": float,      # percentage
    "timestamp":      str,        # ISO 8601
    "source":         str,        # "yfinance" | "finnhub"
  }
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# ── In-memory cache ───────────────────────────────────────────────────────────
# Structure: { symbol: (data_dict, fetched_at_ts) }
_CACHE: dict[str, tuple[dict, float]] = {}
_CACHE_TTL: float = 60.0          # seconds — refresh quote after 60s
_BATCH_CACHE_TTL: float = 120.0   # seconds — batch fetches are heavier


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Finnhub fallback ──────────────────────────────────────────────────────────

def _finnhub_quote(symbol: str) -> dict | None:
    """Fetch real-time quote from Finnhub."""
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    try:
        import finnhub
        client = finnhub.Client(api_key=api_key)
        q = client.quote(symbol.upper())
        if not q or q.get("c", 0) == 0:
            return None
        price     = float(q["c"])
        prev      = float(q["pc"])
        change    = round(price - prev, 4)
        change_pct = round((change / prev) * 100, 4) if prev else 0.0
        return {
            "symbol":         symbol.upper(),
            "price":          price,
            "open":           float(q.get("o", price)),
            "high":           float(q.get("h", price)),
            "low":            float(q.get("l", price)),
            "volume":         int(q.get("v", 0)),
            "avg_volume":     0,
            "change":         change,
            "change_percent": change_pct,
            "timestamp":      _now_iso(),
            "source":         "finnhub",
        }
    except Exception as e:
        logger.warning("Finnhub quote failed for %s: %s", symbol, e)
        return None


# ── yfinance primary ──────────────────────────────────────────────────────────

def _yfinance_quote(symbol: str) -> dict | None:
    """Fetch quote from yfinance."""
    try:
        ticker = yf.Ticker(symbol.upper())
        info   = ticker.fast_info
        price  = float(getattr(info, "last_price", 0) or 0)
        if price == 0:
            hist = ticker.history(period="2d", interval="1d")
            if hist.empty:
                return None
            price     = float(hist["Close"].iloc[-1])
            prev      = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            open_p    = float(hist["Open"].iloc[-1])
            high_p    = float(hist["High"].iloc[-1])
            low_p     = float(hist["Low"].iloc[-1])
            vol       = int(hist["Volume"].iloc[-1])
            avg_vol   = int(hist["Volume"].mean())
        else:
            prev    = float(getattr(info, "previous_close", price) or price)
            open_p  = float(getattr(info, "open", price) or price)
            high_p  = float(getattr(info, "day_high", price) or price)
            low_p   = float(getattr(info, "day_low", price) or price)
            vol     = int(getattr(info, "last_volume", 0) or 0)
            avg_vol = int(getattr(info, "three_month_average_volume", 0) or 0)

        change     = round(price - prev, 4)
        change_pct = round((change / prev) * 100, 4) if prev else 0.0
        return {
            "symbol":         symbol.upper(),
            "price":          round(price, 4),
            "open":           round(open_p, 4),
            "high":           round(high_p, 4),
            "low":            round(low_p, 4),
            "volume":         vol,
            "avg_volume":     avg_vol,
            "change":         change,
            "change_percent": change_pct,
            "timestamp":      _now_iso(),
            "source":         "yfinance",
        }
    except Exception as e:
        logger.warning("yfinance quote failed for %s: %s", symbol, e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_quote(symbol: str, force_refresh: bool = False) -> dict | None:
    """
    Return normalized real-time quote for a single symbol.
    Checks cache first; falls back yfinance → Finnhub.
    """
    sym = symbol.upper()
    now = time.time()

    if not force_refresh and sym in _CACHE:
        data, fetched_at = _CACHE[sym]
        if now - fetched_at < _CACHE_TTL:
            return data

    # Primary: yfinance
    data = _yfinance_quote(sym)
    # Secondary: Finnhub
    if data is None:
        data = _finnhub_quote(sym)
    if data:
        _CACHE[sym] = (data, now)
    return data


def get_quotes_batch(symbols: list[str], force_refresh: bool = False) -> dict[str, dict]:
    """
    Fetch quotes for multiple symbols efficiently using yfinance batch download.
    Returns { symbol: quote_dict }.
    """
    now    = time.time()
    result: dict[str, dict] = {}
    missing: list[str] = []

    for sym in [s.upper() for s in symbols]:
        if not force_refresh and sym in _CACHE:
            data, fetched_at = _CACHE[sym]
            if now - fetched_at < _BATCH_CACHE_TTL:
                result[sym] = data
                continue
        missing.append(sym)

    if not missing:
        return result

    # Batch download via yfinance.
    # Use group_by="column" (default) so columns are (field, ticker) —
    # allowing df["Close"][sym] access. For a single ticker yfinance returns
    # flat columns, handled by the len == 1 branch.
    try:
        df = yf.download(
            missing,
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        for sym in missing:
            try:
                if len(missing) == 1:
                    # Single-ticker: flat columns — Close, Open, High, Low, Volume
                    closes  = df["Close"].dropna()
                    opens   = df["Open"].dropna()
                    highs   = df["High"].dropna()
                    lows    = df["Low"].dropna()
                    volumes = df["Volume"].dropna()
                else:
                    # Multi-ticker with group_by="column":
                    # columns are MultiIndex (field, ticker) — df["Close"][sym]
                    close_df = df["Close"]
                    if sym not in close_df.columns:
                        continue
                    closes  = close_df[sym].dropna()
                    opens   = df["Open"][sym].dropna()
                    highs   = df["High"][sym].dropna()
                    lows    = df["Low"][sym].dropna()
                    volumes = df["Volume"][sym].dropna()

                if closes.empty:
                    continue
                price     = float(closes.iloc[-1])
                prev      = float(closes.iloc[-2]) if len(closes) > 1 else price
                change    = round(price - prev, 4)
                change_pct = round((change / prev) * 100, 4) if prev else 0.0
                quote = {
                    "symbol":         sym,
                    "price":          round(price, 4),
                    "open":           round(float(opens.iloc[-1]), 4) if not opens.empty else round(price, 4),
                    "high":           round(float(highs.iloc[-1]), 4) if not highs.empty else round(price, 4),
                    "low":            round(float(lows.iloc[-1]),  4) if not lows.empty  else round(price, 4),
                    "volume":         int(volumes.iloc[-1]) if not volumes.empty else 0,
                    "avg_volume":     int(volumes.mean())   if not volumes.empty else 0,
                    "change":         change,
                    "change_percent": change_pct,
                    "timestamp":      _now_iso(),
                    "source":         "yfinance_batch",
                }
                _CACHE[sym] = (quote, now)
                result[sym] = quote
            except Exception as exc:
                logger.debug("Batch parse failed for %s: %s", sym, exc)
                continue
    except Exception as e:
        logger.warning("Batch download failed: %s — falling back to individual", e)
        for sym in missing:
            q = get_quote(sym)
            if q:
                result[sym] = q

    return result


def clear_cache(symbol: str | None = None) -> None:
    """Clear cache for a specific symbol or all symbols."""
    if symbol:
        _CACHE.pop(symbol.upper(), None)
    else:
        _CACHE.clear()


def cache_stats() -> dict:
    """Return cache statistics for monitoring."""
    now = time.time()
    return {
        "cached_symbols": len(_CACHE),
        "symbols": [
            {"symbol": sym, "age_seconds": round(now - ts, 1)}
            for sym, (_, ts) in _CACHE.items()
        ],
    }


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Single quote
    q = get_quote("SPY")
    print("Single:", q)

    # Batch quotes
    quotes = get_quotes_batch(["AAPL", "NVDA", "TSLA", "QQQ"])
    for sym, data in quotes.items():
        print(f"{sym}: ${data['price']} ({data['change_percent']:+.2f}%)")
