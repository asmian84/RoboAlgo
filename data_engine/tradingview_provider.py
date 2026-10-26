"""
RoboAlgo — TradingView screener provider (Stage-1 price/volume screen).

LEGAL / OPERATIONAL NOTICE — READ BEFORE ENABLING
-------------------------------------------------
TradingView's Terms of Service prohibit scraping and redistribution of their
data. The endpoint used here (``scanner.tradingview.com/<market>/scan``) is
undocumented and may be rate-limited, changed, or removed without notice.
This module is ONE pluggable provider behind a fallback — it is never the sole
source of data, and every failure path returns ``None`` so the caller falls
back to yfinance. No data is redistributed: results are used in-process for
the operator's own screen and are cached in memory only. Whether to use it
is the operator's decision.

What it does
------------
One server-side filtered query replaces the per-batch yfinance bulk download
in ``scanner_engine.finder._stage1_screen``. Pushing the price/volume band into
the request's ``filter`` narrowed 11,871 US stocks to 719 rows in ~0.23s in a
single call.

Public API
----------
``screen(criteria, timeout=15.0) -> tuple[list[dict], int] | None``

Matches the ``_stage1_screen`` seam: ``(survivors, failed_count)`` where each
survivor is ``{"symbol": str, "price": float, "volume": float}``. Returns
``None`` — never raises — on any failure (network, non-200, malformed JSON,
unexpected shape, incomplete pagination).

*** SAFETY: ONLY ``name``, ``close`` AND ``volume`` ARE EVER REQUESTED ***
------------------------------------------------------------------------
Do NOT add SMA50, SMA200, RSI, change, or any other precomputed indicator
column to ``_COLUMNS``. Those columns are NOT corporate-action-adjusted
against ``close``, and the mismatch is silent and symbol-dependent.

Evidence (re-verified live on 2026-10-06, same request, five symbols):

    symbol  close    SMA50    SMA200   RSI    change
    CTVA    12.39    77.32    77.88    6.38   +3.94%   <-- inconsistent
    SPY    774.83   764.42   720.95   58.83   +0.67%
    MSFT   525.18   490.59   433.07   66.17   +1.48%
    NVDA   238.90   218.92   201.02   66.90   +2.12%
    AAPL   332.89   322.41   289.45   53.79   -0.24%

For NYSE:CTVA the close is 12.39 but SMA50 is 77.32 and SMA200 is 77.88 — the
averages were computed on pre-adjustment prices. A screen reading them would
see CTVA as ~84% below its 50-day mean at RSI 6.4: a fabricated extreme-
oversold signal on a stock that was actually UP 3.9% on the day. The other
four symbols are self-consistent, which is exactly why this is dangerous — it
only bites on whichever symbols happen to have had a corporate action. Acting
on it would be a wrong trade. Indicators are therefore computed downstream
(Stage 2) from adjusted OHLCV history, never taken from this provider.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_SCAN_URL = "https://scanner.tradingview.com/america/scan"
_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

# SAFETY: exactly these three columns. See the module docstring — CTVA's SMA50/
# SMA200/RSI are not adjusted against its close. Do not add indicator columns.
_COLUMNS = ["name", "close", "volume"]

_PAGE_SIZE = 1000
_MAX_PAGES = 20
_PAGE_SLEEP = 0.25  # polite rate limiting between pages

_CACHE_TTL = 900  # 15 min, matches scanner_engine.finder._STAGE1_CACHE_TTL
_CACHE: dict[tuple, tuple[float, list[dict], int]] = {}


def _build_filter(criteria: Any) -> list[dict]:
    flt: list[dict] = [{"left": "type", "operation": "equal", "right": "stock"}]
    lo, hi = criteria.min_price, criteria.max_price
    if lo is not None and hi is not None:
        flt.append({"left": "close", "operation": "in_range", "right": [lo, hi]})
    elif lo is not None:
        flt.append({"left": "close", "operation": "greater", "right": lo})
    elif hi is not None:
        flt.append({"left": "close", "operation": "less", "right": hi})
    flt.append({"left": "volume", "operation": "greater", "right": criteria.min_volume})
    return flt


def _build_body(criteria: Any, offset: int) -> dict:
    return {
        "filter": _build_filter(criteria),
        "options": {"lang": "en"},
        "columns": list(_COLUMNS),
        "sort": {"sortBy": "volume", "sortOrder": "desc"},
        "range": [offset, offset + _PAGE_SIZE],
    }


def _num(value: Any) -> float | None:
    """Finite float, or None for bool / None / strings / NaN / inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _parse_rows(rows: list, criteria: Any) -> tuple[list[dict], int]:
    """Convert raw ``data`` rows to survivors. Returns (survivors, failed).

    ``failed`` counts rows that are malformed or carry a non-numeric / non-
    positive close, or a non-numeric / negative volume. Rows that are valid
    but fall outside the requested band are dropped silently (not "failed"),
    mirroring ``_stage1_screen`` — this also guards against the server
    ignoring part of the filter.
    """
    survivors: list[dict] = []
    failed = 0
    for row in rows:
        try:
            d = row["d"]
            symbol = d[0]  # bare ticker ("WBD"), not row["s"] ("NASDAQ:WBD")
            price = _num(d[1])
            volume = _num(d[2])
        except (KeyError, IndexError, TypeError):
            failed += 1
            continue
        if not isinstance(symbol, str) or not symbol.strip():
            failed += 1
            continue
        if price is None or price <= 0 or volume is None or volume < 0:
            failed += 1
            continue
        if criteria.min_price is not None and price < criteria.min_price:
            continue
        if criteria.max_price is not None and price > criteria.max_price:
            continue
        if volume < criteria.min_volume:
            continue
        survivors.append({"symbol": symbol.strip(), "price": price, "volume": volume})
    return survivors, failed


def _fetch_all(criteria: Any, timeout: float) -> tuple[list[dict], int] | None:
    survivors: list[dict] = []
    failed = 0
    offset = 0
    for page in range(_MAX_PAGES):
        if page:
            time.sleep(_PAGE_SLEEP)
        resp = requests.post(
            _SCAN_URL, json=_build_body(criteria, offset), headers=_HEADERS, timeout=timeout
        )
        if resp.status_code != 200:
            logger.warning("tradingview screen: HTTP %s at offset %d", resp.status_code, offset)
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            logger.warning("tradingview screen: unexpected payload type %s", type(payload).__name__)
            return None
        data = payload.get("data")
        total = payload.get("totalCount")
        if not isinstance(data, list) or isinstance(total, bool) or not isinstance(total, int):
            logger.warning("tradingview screen: unexpected response shape at offset %d", offset)
            return None

        rows_ok, rows_failed = _parse_rows(data, criteria)
        survivors.extend(rows_ok)
        failed += rows_failed
        offset += len(data)

        if len(data) < _PAGE_SIZE or offset >= total:
            return survivors, failed

    # Hit the page cap with rows still outstanding: the result would be silently
    # truncated (lowest-volume names missing). Treat as failure so the caller
    # falls back to a complete source.
    logger.warning("tradingview screen: page cap (%d) reached before totalCount; discarding", _MAX_PAGES)
    return None


def screen(criteria: Any, timeout: float = 15.0) -> tuple[list[dict], int] | None:
    """Server-side price/volume screen of US stocks.

    ``criteria`` is a ``scanner_engine.finder.FinderCriteria`` (only
    ``min_price``, ``max_price`` and ``min_volume`` are read). Returns
    ``(survivors, failed_count)`` or ``None`` on any failure — never raises.
    Results are cached in-process for 15 minutes, keyed on the band.
    """
    try:
        cache_key = (criteria.min_price, criteria.max_price, criteria.min_volume)
        now = time.time()
        cached = _CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _CACHE_TTL:
            return [dict(s) for s in cached[1]], cached[2]

        result = _fetch_all(criteria, timeout)
        if result is None:
            return None
        survivors, failed = result
        _CACHE[cache_key] = (now, survivors, failed)
        return [dict(s) for s in survivors], failed
    except Exception as exc:  # noqa: BLE001 — contract: never raise, caller falls back
        logger.warning("tradingview screen failed: %s: %s", type(exc).__name__, exc)
        return None
