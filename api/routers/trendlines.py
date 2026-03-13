"""
RoboAlgo - Trendline Detection Engine
Identifies swing highs/lows from price data and returns:
  - Resistance trendlines (connecting swing highs)
  - Support trendlines (connecting swing lows)
  - Projected extensions to current date
Across three sensitivity windows: short (5), medium (10), long (20).

Works for ANY ticker: DB path for known instruments, yfinance fallback
for everything else.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter
from sqlalchemy import select

from database.connection import get_session
from database.models import Instrument, PriceData

router = APIRouter()


# ── Pure-Python trendline math ────────────────────────────────────────────────

def _find_pivots(
    dates: list[date],
    highs: list[float],
    lows:  list[float],
    window: int,
) -> tuple[list[dict], list[dict]]:
    """
    Return swing highs and swing lows using a symmetric rolling window.
    A swing high is a bar whose high is greater than all bars within ±window.
    A swing low is a bar whose low is less than all bars within ±window.
    """
    n = len(dates)
    swing_highs: list[dict] = []
    swing_lows:  list[dict] = []

    for i in range(window, n - window):
        is_high = all(
            highs[i] > highs[j]
            for j in range(i - window, i + window + 1)
            if j != i
        )
        if is_high:
            swing_highs.append({"date": dates[i].isoformat(), "value": round(highs[i], 4)})

        is_low = all(
            lows[i] < lows[j]
            for j in range(i - window, i + window + 1)
            if j != i
        )
        if is_low:
            swing_lows.append({"date": dates[i].isoformat(), "value": round(lows[i], 4)})

    return swing_highs, swing_lows


def _project_line(p1: dict, p2: dict, target_date: date) -> float | None:
    """
    Linear extrapolation from two pivot points to a target date.
    Returns None if the two points share the same date.
    """
    d1   = date.fromisoformat(p1["date"])
    d2   = date.fromisoformat(p2["date"])
    span = (d2 - d1).days
    if span == 0:
        return None
    slope = (p2["value"] - p1["value"]) / span
    return round(p1["value"] + slope * (target_date - d1).days, 4)


def _build_trendlines(pivots: list[dict], pivot_type: str, latest_date: date) -> list[dict]:
    """
    Connect consecutive pivot pairs into trendline segments.
    Projects each line to latest_date.
    """
    lines: list[dict] = []
    for i in range(len(pivots) - 1):
        p1, p2 = pivots[i], pivots[i + 1]
        projected = _project_line(p1, p2, latest_date)
        line: dict = {"type": pivot_type, "points": [p1, p2]}
        if projected is not None and latest_date.isoformat() > p2["date"]:
            line["projected"] = {"date": latest_date.isoformat(), "value": projected}
        lines.append(line)
    return lines


def _compute_trendlines(
    dates: list[date],
    highs: list[float],
    lows:  list[float],
    symbol: str,
) -> dict:
    """
    Run the three-window trendline analysis on raw dates/highs/lows lists.
    Returns the fully-formed response dict.
    """
    latest = dates[-1]
    result: dict = {"symbol": symbol}

    for label, window, lookback in [
        ("short",  5,  90),
        ("medium", 10, 365),
        ("long",   20, len(dates)),
    ]:
        sd = dates[-lookback:]
        sh = highs[-lookback:]
        sl = lows[-lookback:]

        swing_h, swing_l = _find_pivots(sd, sh, sl, window)

        result[label] = {
            "window":         window,
            "lookback_bars":  min(lookback, len(dates)),
            "swing_highs":    swing_h,
            "swing_lows":     swing_l,
            "resistance":     _build_trendlines(swing_h, "resistance", latest),
            "support":        _build_trendlines(swing_l, "support",    latest),
        }

    return result


def _yfinance_fetch_prices(symbol: str) -> tuple[list[date], list[float], list[float]]:
    """
    Fetch 2 years of daily OHLCV via yfinance.
    Returns (dates, highs, lows) — three empty lists on failure.
    """
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)
        if hist.empty:
            return [], [], []

        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)

        return (
            [ts.date() for ts in hist.index],
            hist["High"].tolist(),
            hist["Low"].tolist(),
        )
    except Exception:
        return [], [], []


def _alphavantage_fetch_prices(symbol: str) -> tuple[list[date], list[float], list[float]]:
    """
    Alpha Vantage TIME_SERIES_DAILY_ADJUSTED fallback for trendlines.
    Returns (dates, highs, lows) — three empty lists on failure.
    """
    try:
        import httpx

        with httpx.Client(timeout=10) as client:
            r = client.get("https://www.alphavantage.co/query", params={
                "function":   "TIME_SERIES_DAILY_ADJUSTED",
                "symbol":     symbol,
                "outputsize": "full",
                "apikey":     "UPSSIQEAMIJFI6RS",
            })

        ts = r.json().get("Time Series (Daily)")
        if not ts:
            return [], [], []

        sorted_items = sorted(ts.items())
        dates = [date.fromisoformat(d) for d, _ in sorted_items]
        highs = [float(v["2. high"]) for _, v in sorted_items]
        lows  = [float(v["3. low"])  for _, v in sorted_items]

        return dates, highs, lows
    except Exception:
        return [], [], []


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/{symbol}")
def get_trendlines(symbol: str):
    """
    Detect support and resistance trendlines from historical price data.
    Returns three sensitivity levels:
      short  (window=5)  — intraday / day-trade perspective
      medium (window=10) — swing-trade perspective
      long   (window=20) — position / weekly perspective

    DB path used for known instruments; yfinance fallback for all others.
    """
    symbol = symbol.upper()
    session = get_session()
    dates: list[date] = []
    highs: list[float] = []
    lows:  list[float] = []

    try:
        instr = session.execute(
            select(Instrument).where(Instrument.symbol == symbol)
        ).scalar_one_or_none()

        if instr:
            # ── DB path ────────────────────────────────────────────────────
            rows = list(session.execute(
                select(PriceData.date, PriceData.high, PriceData.low)
                .where(PriceData.instrument_id == instr.id)
                .where(PriceData.high.isnot(None))
                .where(PriceData.low.isnot(None))
                .order_by(PriceData.date)
            ).all())

            if len(rows) >= 50:
                dates = [r.date for r in rows]
                highs = [r.high for r in rows]
                lows  = [r.low  for r in rows]

    finally:
        session.close()

    # ── yfinance fallback ──────────────────────────────────────────────────
    if len(dates) < 50:
        dates, highs, lows = _yfinance_fetch_prices(symbol)

    # ── Alpha Vantage final fallback ───────────────────────────────────────
    if len(dates) < 50:
        dates, highs, lows = _alphavantage_fetch_prices(symbol)

    if len(dates) < 50:
        return {"symbol": symbol, "error": "Insufficient price data"}

    return _compute_trendlines(dates, highs, lows, symbol)
