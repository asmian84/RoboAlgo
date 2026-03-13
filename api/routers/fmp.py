"""
Financial Modeling Prep (FMP) — Stable API client
Provides: real-time quotes, company profiles, earnings, analyst targets, financial ratios

All functions use a simple TTL cache (per-symbol) to avoid hammering the API.
API key is stored here; callers just call the helper functions.
"""
from __future__ import annotations

import time
import urllib.request
import json
from typing import Any

from fastapi import APIRouter

router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
FMP_KEY  = "V981vaUJOuOmNrltZDCKNt3RSwsc0MGW"
FMP_BASE = "https://financialmodelingprep.com/stable"

# ── TTL caches (keyed by symbol) ──────────────────────────────────────────────
_quote_cache:    dict[str, tuple[dict, float]] = {}
_profile_cache:  dict[str, tuple[dict, float]] = {}
_earnings_cache: dict[str, tuple[list, float]] = {}
_targets_cache:  dict[str, tuple[dict, float]] = {}
_ratios_cache:   dict[str, tuple[dict, float]] = {}

_QUOTE_TTL    = 60          #  1 min  — live prices
_PROFILE_TTL  = 24 * 3600   # 24 hrs  — rarely changes
_EARNINGS_TTL = 4  * 3600   #  4 hrs  — next earnings date
_TARGETS_TTL  = 4  * 3600   #  4 hrs  — analyst targets
_RATIOS_TTL   = 24 * 3600   # 24 hrs  — annual ratios


def _get(path: str, timeout: int = 8) -> Any:
    """HTTP GET to FMP, returns parsed JSON or raises."""
    url = f"{FMP_BASE}{path}&apikey={FMP_KEY}" if "?" in path else f"{FMP_BASE}{path}?apikey={FMP_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "RoboAlgo/3"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_quote(symbol: str) -> dict | None:
    """Real-time quote: price, change, volume, market-cap, 52w range, MA50/200."""
    now = time.time()
    if symbol in _quote_cache:
        data, ts = _quote_cache[symbol]
        if now - ts < _QUOTE_TTL:
            return data
    try:
        result = _get(f"/quote?symbol={symbol}")
        if result and isinstance(result, list):
            _quote_cache[symbol] = (result[0], now)
            return result[0]
    except Exception:
        pass
    return _quote_cache.get(symbol, (None, 0))[0]  # return stale on error


def get_profile(symbol: str) -> dict | None:
    """Company profile: sector, industry, description, CEO, employees, beta, IPO date."""
    now = time.time()
    if symbol in _profile_cache:
        data, ts = _profile_cache[symbol]
        if now - ts < _PROFILE_TTL:
            return data
    try:
        result = _get(f"/profile?symbol={symbol}")
        if result and isinstance(result, list):
            _profile_cache[symbol] = (result[0], now)
            return result[0]
    except Exception:
        pass
    return _profile_cache.get(symbol, (None, 0))[0]


def get_earnings(symbol: str) -> list[dict]:
    """Earnings calendar: date, epsActual, epsEstimated, revenueActual, revenueEstimated."""
    now = time.time()
    if symbol in _earnings_cache:
        data, ts = _earnings_cache[symbol]
        if now - ts < _EARNINGS_TTL:
            return data
    try:
        result = _get(f"/earnings?symbol={symbol}")
        if isinstance(result, list):
            _earnings_cache[symbol] = (result, now)
            return result
    except Exception:
        pass
    return _earnings_cache.get(symbol, ([], 0))[0]


def get_next_earnings(symbol: str) -> dict | None:
    """Returns the next (upcoming) earnings entry, or None."""
    from datetime import date
    today = date.today().isoformat()
    earnings = get_earnings(symbol)
    upcoming = [e for e in earnings if e.get("date", "") >= today and e.get("epsActual") is None]
    return upcoming[0] if upcoming else None


def get_price_targets(symbol: str) -> dict | None:
    """Analyst price target summary: avg target last month/quarter/year + count."""
    now = time.time()
    if symbol in _targets_cache:
        data, ts = _targets_cache[symbol]
        if now - ts < _TARGETS_TTL:
            return data
    try:
        result = _get(f"/price-target-summary?symbol={symbol}")
        if result and isinstance(result, list):
            _targets_cache[symbol] = (result[0], now)
            return result[0]
    except Exception:
        pass
    return _targets_cache.get(symbol, (None, 0))[0]


def get_ratios(symbol: str) -> dict | None:
    """Latest annual financial ratios: P/E, margins, dividend yield, etc."""
    now = time.time()
    if symbol in _ratios_cache:
        data, ts = _ratios_cache[symbol]
        if now - ts < _RATIOS_TTL:
            return data
    try:
        result = _get(f"/ratios?symbol={symbol}&period=annual&limit=1")
        if result and isinstance(result, list):
            _ratios_cache[symbol] = (result[0], now)
            return result[0]
    except Exception:
        pass
    return _ratios_cache.get(symbol, (None, 0))[0]


# ── FastAPI endpoint ──────────────────────────────────────────────────────────

@router.get("/{symbol}/fundamentals")
def get_fundamentals(symbol: str):
    """
    Aggregated fundamentals from FMP for the chart sidebar.
    Returns: company info, valuation metrics, analyst targets, next earnings.
    """
    sym = symbol.upper()

    profile  = get_profile(sym)
    ratios   = get_ratios(sym)
    targets  = get_price_targets(sym)
    quote    = get_quote(sym)
    next_e   = get_next_earnings(sym)

    def _pct(v):
        return round(v * 100, 1) if v is not None else None

    def _round(v, n=2):
        return round(v, n) if v is not None else None

    return {
        "symbol": sym,
        # Company
        "company_name":  profile.get("companyName")       if profile else None,
        "sector":        profile.get("sector")             if profile else None,
        "industry":      profile.get("industry")           if profile else None,
        "description":   (profile.get("description") or "")[:280] if profile else None,
        "ceo":           profile.get("ceo")                if profile else None,
        "employees":     profile.get("fullTimeEmployees")  if profile else None,
        "website":       profile.get("website")            if profile else None,
        "ipo_date":      profile.get("ipoDate")            if profile else None,
        "is_etf":        profile.get("isEtf", False)       if profile else False,
        "beta":          _round(profile.get("beta"), 3)    if profile else None,
        # Market data
        "market_cap":    quote.get("marketCap")            if quote else (profile.get("marketCap") if profile else None),
        "price":         quote.get("price")                if quote else None,
        "year_high":     quote.get("yearHigh")             if quote else None,
        "year_low":      quote.get("yearLow")              if quote else None,
        "price_avg_50":  quote.get("priceAvg50")           if quote else None,
        "price_avg_200": quote.get("priceAvg200")          if quote else None,
        # Valuation
        "pe_ratio":           _round(ratios.get("priceToEarningsRatio"), 1)    if ratios else None,
        "peg_ratio":          _round(ratios.get("priceToEarningsGrowthRatio"), 2) if ratios else None,
        "ps_ratio":           _round(ratios.get("priceToSalesRatio"), 1)       if ratios else None,
        "pb_ratio":           _round(ratios.get("priceToBookRatio"), 1)        if ratios else None,
        "ev_ebitda":          _round(ratios.get("enterpriseValueMultiple"), 1) if ratios else None,
        # Profitability
        "gross_margin":       _pct(ratios.get("grossProfitMargin"))            if ratios else None,
        "operating_margin":   _pct(ratios.get("operatingProfitMargin"))       if ratios else None,
        "net_margin":         _pct(ratios.get("netProfitMargin"))              if ratios else None,
        "free_cashflow_margin": _pct(ratios.get("freeCashFlowOperatingCashFlowRatio")) if ratios else None,
        # Per-share
        "eps":            _round(ratios.get("netIncomePerShare"), 2)           if ratios else None,
        "fcf_per_share":  _round(ratios.get("freeCashFlowPerShare"), 2)       if ratios else None,
        "book_per_share": _round(ratios.get("bookValuePerShare"), 2)          if ratios else None,
        "dividend_yield": _pct(ratios.get("dividendYield"))                   if ratios else None,
        # Analyst targets
        "analyst_target_1m":  targets.get("lastMonthAvgPriceTarget")          if targets else None,
        "analyst_target_1q":  targets.get("lastQuarterAvgPriceTarget")        if targets else None,
        "analyst_count_1q":   targets.get("lastQuarterCount")                 if targets else None,
        # Earnings
        "next_earnings_date":  next_e.get("date")          if next_e else None,
        "next_eps_estimate":   next_e.get("epsEstimated")  if next_e else None,
        "next_rev_estimate":   next_e.get("revenueEstimated") if next_e else None,
    }
