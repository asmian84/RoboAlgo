"""
lab_engine/fundamentals.py
────────────────────────────────────────────────────────────────────────────────
Switchable fundamentals provider for the Lab.

Two backends, same normalized output contract:
  "yfinance"          — free, no key, whatever .info exposes
  "financialdatasets" — Financial Datasets API (what ai-hedge-fund uses),
                        needs FINANCIAL_DATASETS_API_KEY

Callers pick per-request; nothing here is wired into the live scanner.

Normalized contract — every key always present, None when unavailable:
  price, market_cap, sector, beta,
  pe_ratio, pb_ratio, ps_ratio,
  roe, roic,
  debt_to_equity, current_ratio,
  gross_margin, operating_margin, net_margin, fcf_margin,
  revenue_growth, earnings_growth,
  dividend_yield, short_percent_float,
  week52_high, week52_low,
  _source, _fields_present
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

PROVIDERS = ("yfinance", "financialdatasets")
DEFAULT_PROVIDER = os.getenv("LAB_FUNDAMENTALS_PROVIDER", "yfinance")

_CACHE: dict[str, tuple[dict, float]] = {}
_TTL = 900.0  # 15 min — fundamentals barely move intraday

_FIELDS = (
    "price", "market_cap", "sector", "beta",
    "pe_ratio", "pb_ratio", "ps_ratio",
    "roe", "roic",
    "debt_to_equity", "current_ratio",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin",
    "revenue_growth", "earnings_growth",
    "dividend_yield", "short_percent_float",
    "week52_high", "week52_low",
)


def _blank() -> dict[str, Any]:
    return {f: None for f in _FIELDS}


def _f(v: Any) -> float | None:
    """Coerce to float, mapping junk/zero-ish sentinels to None."""
    try:
        if v is None:
            return None
        out = float(v)
        if out != out:  # NaN
            return None
        return out
    except (TypeError, ValueError):
        return None


# ── yfinance ──────────────────────────────────────────────────────────────

def _from_yfinance(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    data = _blank()
    t = yf.Ticker(symbol.upper())
    info = t.info or {}

    data["price"]        = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
    data["market_cap"]   = _f(info.get("marketCap"))
    data["sector"]       = info.get("sector")
    data["beta"]         = _f(info.get("beta"))
    data["pe_ratio"]     = _f(info.get("trailingPE"))
    data["pb_ratio"]     = _f(info.get("priceToBook"))
    data["ps_ratio"]     = _f(info.get("priceToSalesTrailing12Months"))
    data["roe"]          = _f(info.get("returnOnEquity"))
    data["roic"]         = _f(info.get("returnOnAssets"))  # proxy — yf has no ROIC
    data["current_ratio"]     = _f(info.get("currentRatio"))
    data["gross_margin"]      = _f(info.get("grossMargins"))
    data["operating_margin"]  = _f(info.get("operatingMargins"))
    data["net_margin"]        = _f(info.get("profitMargins"))
    data["revenue_growth"]    = _f(info.get("revenueGrowth"))
    data["earnings_growth"]   = _f(info.get("earningsGrowth")) or _f(info.get("earningsQuarterlyGrowth"))
    data["dividend_yield"]      = _f(info.get("dividendYield"))
    data["short_percent_float"] = _f(info.get("shortPercentOfFloat"))
    data["week52_high"] = _f(info.get("fiftyTwoWeekHigh"))
    data["week52_low"]  = _f(info.get("fiftyTwoWeekLow"))

    # yfinance reports debtToEquity as a percentage (e.g. 145.0 = 1.45x)
    dte = _f(info.get("debtToEquity"))
    data["debt_to_equity"] = dte / 100.0 if dte is not None and dte > 5 else dte

    fcf, mcap = _f(info.get("freeCashflow")), data["market_cap"]
    rev = _f(info.get("totalRevenue"))
    if fcf is not None and rev:
        data["fcf_margin"] = fcf / rev

    return data


# ── Financial Datasets API ────────────────────────────────────────────────

def _from_financialdatasets(symbol: str) -> dict[str, Any]:
    key = os.getenv("FINANCIAL_DATASETS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FINANCIAL_DATASETS_API_KEY not set — add it to .env or use provider=yfinance"
        )

    data = _blank()
    headers = {"X-API-KEY": key}
    base = "https://api.financialdatasets.ai"

    r = requests.get(
        f"{base}/financial-metrics/snapshot",
        params={"ticker": symbol.upper()},
        headers=headers,
        timeout=12,
    )
    r.raise_for_status()
    m = (r.json() or {}).get("snapshot") or {}

    data["price"]        = _f(m.get("market_cap") and m.get("price"))
    data["market_cap"]   = _f(m.get("market_cap"))
    data["pe_ratio"]     = _f(m.get("price_to_earnings_ratio"))
    data["pb_ratio"]     = _f(m.get("price_to_book_ratio"))
    data["ps_ratio"]     = _f(m.get("price_to_sales_ratio"))
    data["roe"]          = _f(m.get("return_on_equity"))
    data["roic"]         = _f(m.get("return_on_invested_capital"))
    data["debt_to_equity"]   = _f(m.get("debt_to_equity"))
    data["current_ratio"]    = _f(m.get("current_ratio"))
    data["gross_margin"]     = _f(m.get("gross_margin"))
    data["operating_margin"] = _f(m.get("operating_margin"))
    data["net_margin"]       = _f(m.get("net_margin"))
    data["fcf_margin"]       = _f(m.get("free_cash_flow_margin"))
    data["revenue_growth"]   = _f(m.get("revenue_growth"))
    data["earnings_growth"]  = _f(m.get("earnings_growth"))

    # Snapshot has no price/sector/52w — backfill from yfinance so personas
    # that need price context still work under this provider.
    try:
        yf_data = _from_yfinance(symbol)
        for k in ("price", "sector", "beta", "dividend_yield",
                  "short_percent_float", "week52_high", "week52_low"):
            if data.get(k) is None:
                data[k] = yf_data.get(k)
    except Exception as exc:
        logger.debug("yfinance backfill failed for %s: %s", symbol, exc)

    return data


# ── Public API ────────────────────────────────────────────────────────────

def get_fundamentals(
    symbol: str,
    provider: str = DEFAULT_PROVIDER,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Fetch normalized fundamentals for `symbol` from the chosen provider.

    Never raises on data problems — returns the blank contract with
    `_fields_present: 0` so personas can degrade to "insufficient data".
    An unknown provider name IS an error (programmer mistake, not a data gap).
    """
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r} — expected one of {PROVIDERS}")

    sym = symbol.upper()
    cache_key = f"{provider}:{sym}"
    now = time.time()
    if not force_refresh and cache_key in _CACHE:
        cached, ts = _CACHE[cache_key]
        if now - ts < _TTL:
            return dict(cached)

    try:
        data = (_from_yfinance if provider == "yfinance" else _from_financialdatasets)(sym)
        data["_error"] = None
    except Exception as exc:
        logger.warning("fundamentals[%s/%s] failed: %s", provider, sym, exc)
        data = _blank()
        data["_error"] = str(exc)

    data["_source"] = provider
    data["_symbol"] = sym
    data["_fields_present"] = sum(1 for f in _FIELDS if data.get(f) is not None)
    _CACHE[cache_key] = (data, now)
    return dict(data)


def provider_status() -> list[dict[str, Any]]:
    """Which providers are usable right now — drives the UI toggle."""
    return [
        {
            "id": "yfinance",
            "label": "yfinance (free)",
            "available": True,
            "note": "No key required. Basic ratios only.",
        },
        {
            "id": "financialdatasets",
            "label": "Financial Datasets API",
            "available": bool(os.getenv("FINANCIAL_DATASETS_API_KEY", "")),
            "note": "Full statement metrics. Needs FINANCIAL_DATASETS_API_KEY in .env.",
        },
    ]
