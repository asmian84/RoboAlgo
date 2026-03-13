"""
Alpha Vantage integration — News Sentiment and Earnings Calendar.
Provides:
  - GET /api/av/sentiment/{symbol}   → numeric sentiment score + article count
  - GET /api/av/earnings/{symbol}    → next earnings date + days-until
  - get_news_sentiment_score(symbol) → float (-1 to 1) for internal use
  - get_earnings_date(symbol)        → date | None for internal use
"""

import os
import time
from datetime import date

import httpx
from fastapi import APIRouter

# Read from env so premium key can be set in .env without changing code.
# Falls back to the bundled free-tier key if not overridden.
AV_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "UPSSIQEAMIJFI6RS")
AV_BASE    = "https://www.alphavantage.co/query"

# ── In-memory TTL caches (survive per-process lifetime) ──────────────────────
_sentiment_cache: dict[str, tuple[float | None, int, float]] = {}
#  symbol → (avg_score, article_count, timestamp)

_earnings_cache: dict[str, tuple[date | None, float]] = {}
#  symbol → (next_earnings_date, timestamp)

_SENTIMENT_TTL = 3_600    # 1 hour
_EARNINGS_TTL  = 86_400   # 24 hours

router = APIRouter()


# ── Internal helpers (called by recommendation.py) ────────────────────────────

def get_news_sentiment_score(symbol: str) -> tuple[float | None, int]:
    """
    Returns (avg_ticker_sentiment_score, article_count) for symbol.
    Score range: -1.0 (very negative) to +1.0 (very positive).
    Returns (None, 0) if unavailable (rate-limit, no data, error).
    """
    now = time.time()
    cached = _sentiment_cache.get(symbol)
    if cached and now - cached[2] < _SENTIMENT_TTL:
        return cached[0], cached[1]

    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(
                AV_BASE,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": symbol,
                    "limit": 50,
                    "apikey": AV_API_KEY,
                },
            )
        data = r.json()

        # Rate-limit check
        if "Note" in data or "Information" in data:
            # Don't cache — try again next request
            return None, 0

        feed = data.get("feed", [])
        scores: list[float] = []
        for article in feed:
            for ts_item in article.get("ticker_sentiment", []):
                if ts_item.get("ticker", "").upper() == symbol.upper():
                    try:
                        scores.append(float(ts_item["ticker_sentiment_score"]))
                    except (KeyError, ValueError):
                        pass

        if not scores:
            _sentiment_cache[symbol] = (None, 0, now)
            return None, 0

        avg = round(sum(scores) / len(scores), 4)
        _sentiment_cache[symbol] = (avg, len(scores), now)
        return avg, len(scores)

    except Exception:
        return None, 0


def get_earnings_date(symbol: str) -> date | None:
    """
    Returns the next upcoming earnings report date for symbol, or None.
    Primary: FMP stable API (reliable, no rate-limit).
    Fallback: Alpha Vantage EARNINGS_CALENDAR (CSV).
    """
    # ── Primary: FMP ─────────────────────────────────────────────────────────
    try:
        from api.routers.fmp import get_next_earnings
        next_e = get_next_earnings(symbol.upper())
        if next_e and next_e.get("date"):
            return date.fromisoformat(next_e["date"])
    except Exception:
        pass

    # ── Fallback: Alpha Vantage ───────────────────────────────────────────────
    now = time.time()
    cached = _earnings_cache.get(symbol)
    if cached and now - cached[1] < _EARNINGS_TTL:
        return cached[0]

    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(
                AV_BASE,
                params={
                    "function": "EARNINGS_CALENDAR",
                    "symbol": symbol,
                    "horizon": "3month",
                    "apikey": AV_API_KEY,
                },
            )
        lines = r.text.strip().split("\n")
        today = date.today()
        result: date | None = None
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[0].strip().upper() == symbol.upper():
                try:
                    dt = date.fromisoformat(parts[2].strip())
                    if dt >= today:
                        result = dt
                        break
                except ValueError:
                    pass
        _earnings_cache[symbol] = (result, now)
        return result
    except Exception:
        _earnings_cache[symbol] = (None, now)
        return None


# ── FastAPI endpoints (for direct frontend queries or debugging) ──────────────

@router.get("/sentiment/{symbol}")
def av_sentiment(symbol: str):
    """Alpha Vantage news sentiment for a ticker symbol."""
    symbol = symbol.upper()
    score, count = get_news_sentiment_score(symbol)
    label = (
        "BULLISH"  if score is not None and score >  0.25 else
        "BEARISH"  if score is not None and score < -0.25 else
        "NEUTRAL"  if score is not None else
        "NO DATA"
    )
    color = (
        "#22c55e" if label == "BULLISH" else
        "#ef4444" if label == "BEARISH" else
        "#9ca3af"
    )
    return {
        "symbol": symbol,
        "sentiment_score": score,
        "article_count": count,
        "label": label,
        "color": color,
    }


@router.get("/earnings/{symbol}")
def av_earnings(symbol: str):
    """Next upcoming earnings date for a ticker symbol."""
    symbol = symbol.upper()
    earnings_date = get_earnings_date(symbol)
    today = date.today()
    days_until = (earnings_date - today).days if earnings_date else None
    within_5d  = days_until is not None and days_until <= 5
    return {
        "symbol":       symbol,
        "earnings_date": earnings_date.isoformat() if earnings_date else None,
        "days_until":   days_until,
        "earnings_risk": within_5d,
    }
