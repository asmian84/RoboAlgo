"""
context_engine/catalyst_engine.py
────────────────────────────────────────────────────────────────────────────────
News and catalyst scoring engine.

Sources (in priority order):
  1. Finnhub company news API (free tier, key from env)
  2. yfinance news endpoint (free, no key)
  3. RSS fallback placeholder (extensible)

Catalyst types detected:
  earnings        → EPS beat/miss, revenue surprise
  guidance        → raised/lowered guidance
  upgrade         → analyst upgrade
  downgrade       → analyst downgrade
  news_spike      → unusual news volume
  merger_acq      → M&A activity
  fda_approval    → biotech catalysts
  insider_buy     → insider purchase filing
  short_squeeze   → short interest context
  none            → no material catalyst

Output per symbol:
  {
    "symbol":         str,
    "catalyst_score": float,     # 0.0 – 1.0 (higher = stronger catalyst)
    "catalyst_type":  str,       # primary catalyst
    "catalyst_list":  list[str], # all detected catalysts
    "sentiment":      str,       # "bullish" | "bearish" | "neutral"
    "headline":       str,       # most relevant headline
    "source":         str,
  }
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Keyword maps ──────────────────────────────────────────────────────────────

_CATALYST_KEYWORDS: dict[str, list[str]] = {
    "earnings":     ["earnings", "eps", "quarterly", "revenue", "profit", "loss", "beat", "miss", "q1", "q2", "q3", "q4"],
    "guidance":     ["guidance", "outlook", "forecast", "raised", "lowered", "raised guidance", "cuts outlook"],
    "upgrade":      ["upgrade", "outperform", "buy rating", "overweight", "price target raised"],
    "downgrade":    ["downgrade", "underperform", "sell rating", "underweight", "price target cut", "price target lowered"],
    "merger_acq":   ["merger", "acquisition", "acquire", "takeover", "buyout", "deal", "agreement"],
    "fda_approval": ["fda", "approval", "approved", "clearance", "clinical trial", "phase 3", "drug"],
    "insider_buy":  ["insider", "insider buy", "purchased shares", "10-b5"],
    "short_squeeze": ["short squeeze", "short interest", "heavily shorted", "gamma squeeze"],
    "news_spike":   [],  # detected by volume of articles, not keywords
}

_BULLISH_KEYWORDS = [
    "beat", "exceeds", "raised", "upgrade", "outperform", "buy", "strong",
    "record", "growth", "approval", "deal", "positive", "surge", "rally",
]
_BEARISH_KEYWORDS = [
    "miss", "below", "lowered", "downgrade", "underperform", "sell", "weak",
    "cut", "loss", "decline", "warning", "risk", "drop", "fell", "concern",
]

# Catalyst weight table — how much each catalyst contributes to the score
_CATALYST_WEIGHTS: dict[str, float] = {
    "earnings":      0.90,
    "guidance":      0.75,
    "merger_acq":    0.85,
    "fda_approval":  0.90,
    "upgrade":       0.60,
    "downgrade":     0.60,
    "insider_buy":   0.50,
    "short_squeeze": 0.70,
    "news_spike":    0.40,
    "none":          0.05,
}


def _classify_headline(text: str) -> tuple[str, float, str]:
    """
    Classify a single headline.
    Returns (catalyst_type, score_contribution, sentiment).
    """
    text_lower = text.lower()
    detected: list[tuple[str, float]] = []

    for cat, keywords in _CATALYST_KEYWORDS.items():
        if cat == "news_spike":
            continue
        if any(kw in text_lower for kw in keywords):
            detected.append((cat, _CATALYST_WEIGHTS[cat]))

    if not detected:
        return "none", 0.05, "neutral"

    # Pick highest-weight catalyst
    best_cat, best_weight = max(detected, key=lambda x: x[1])

    # Sentiment
    bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in text_lower)
    sentiment = "bullish" if bull > bear else ("bearish" if bear > bull else "neutral")

    return best_cat, best_weight, sentiment


def _fetch_finnhub_news(symbol: str, days_back: int = 3) -> list[dict]:
    """Fetch news from Finnhub company news endpoint."""
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return []
    try:
        import finnhub
        client   = finnhub.Client(api_key=api_key)
        date_to  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date_from = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        news     = client.company_news(symbol.upper(), _from=date_from, to=date_to)
        return news or []
    except Exception as e:
        logger.debug("Finnhub news failed for %s: %s", symbol, e)
        return []


def _fetch_yfinance_news(symbol: str) -> list[dict]:
    """Fetch news from yfinance (no API key required)."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        news   = ticker.news or []
        return [
            {
                "headline": n.get("title", ""),
                "summary":  n.get("summary", ""),
                "datetime": n.get("providerPublishTime", 0),
            }
            for n in news
        ]
    except Exception as e:
        logger.debug("yfinance news failed for %s: %s", symbol, e)
        return []


class CatalystEngine:
    """Score news catalysts for a list of symbols."""

    def __init__(
        self,
        days_back:       int   = 3,
        news_spike_min:  int   = 5,   # articles in last 24h to flag news_spike
        min_score:       float = 0.0,
    ):
        self.days_back      = days_back
        self.news_spike_min = news_spike_min
        self.min_score      = min_score

    def score_symbol(self, symbol: str) -> dict[str, Any]:
        """Score catalyst strength for a single symbol."""
        sym = symbol.upper()

        # Collect headlines from all sources
        headlines: list[str] = []

        # Try Finnhub first
        fh_news = _fetch_finnhub_news(sym, self.days_back)
        for item in fh_news:
            h = item.get("headline", "") or item.get("summary", "")
            if h:
                headlines.append(h)

        # Fallback / supplement with yfinance
        if not headlines:
            yf_news = _fetch_yfinance_news(sym)
            for item in yf_news:
                h = item.get("headline", "")
                if h:
                    headlines.append(h)

        if not headlines:
            return self._empty(sym)

        # Score each headline
        scores:     list[float] = []
        catalysts:  list[str]   = []
        sentiments: list[str]   = []
        top_headline = headlines[0]
        top_score    = 0.0

        for hl in headlines:
            cat, sc, sent = _classify_headline(hl)
            scores.append(sc)
            catalysts.append(cat)
            sentiments.append(sent)
            if sc > top_score:
                top_score    = sc
                top_headline = hl

        # News spike detection
        if len(headlines) >= self.news_spike_min:
            catalysts.append("news_spike")
            scores.append(_CATALYST_WEIGHTS["news_spike"])

        # Aggregate: take max score + small bonus for multiple catalysts
        unique_cats  = list(dict.fromkeys(c for c in catalysts if c != "none"))
        base_score   = max(scores) if scores else 0.0
        multi_bonus  = min(0.10, len(unique_cats) * 0.03)
        final_score  = min(1.0, base_score + multi_bonus)

        # Aggregate sentiment
        bull = sentiments.count("bullish")
        bear = sentiments.count("bearish")
        sentiment = "bullish" if bull > bear else ("bearish" if bear > bull else "neutral")

        return {
            "symbol":        sym,
            "catalyst_score": round(final_score, 3),
            "catalyst_type":  unique_cats[0] if unique_cats else "none",
            "catalyst_list":  unique_cats,
            "sentiment":      sentiment,
            "headline":       top_headline[:200],
            "article_count":  len(headlines),
            "source":         "finnhub" if fh_news else "yfinance",
        }

    def score_batch(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Score catalysts for multiple symbols. Returns sorted by score desc."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(self.score_symbol, sym): sym for sym in symbols}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r["catalyst_score"] >= self.min_score:
                        results.append(r)
                except Exception:
                    continue
        results.sort(key=lambda x: x["catalyst_score"], reverse=True)
        return results

    @staticmethod
    def _empty(symbol: str) -> dict[str, Any]:
        return {
            "symbol":         symbol,
            "catalyst_score": 0.05,
            "catalyst_type":  "none",
            "catalyst_list":  [],
            "sentiment":      "neutral",
            "headline":       "",
            "article_count":  0,
            "source":         "none",
        }


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = CatalystEngine(days_back=3)
    symbols = ["AAPL", "NVDA", "TSLA"]
    for sym in symbols:
        result = engine.score_symbol(sym)
        print(f"{result['symbol']:6} | score={result['catalyst_score']:.2f} | "
              f"type={result['catalyst_type']:12} | sentiment={result['sentiment']} | "
              f"{result['headline'][:60]}")
